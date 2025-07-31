import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Check availability of optional foundation model libraries
try:
    import open_clip
    from open_clip import create_model_from_pretrained, get_tokenizer
    OPEN_CLIP_AVAILABLE = True
    print("OpenCLIP available")
except ImportError:
    print("OpenCLIP not available, install with: pip install open-clip-torch")
    OPEN_CLIP_AVAILABLE = False

try:
    from transformers import CLIPModel, CLIPProcessor, ViTForImageClassification, ViTImageProcessor
    TRANSFORMERS_AVAILABLE = True
    print("Transformers available")
except ImportError:
    print("Transformers not available")
    TRANSFORMERS_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
    print("  TIMM available")
except ImportError:
    print(" TIMM not available")
    TIMM_AVAILABLE = False

def detect_available_foundation_models():
    """Detect available foundation models on the system."""
    available = {}
    # Check BiomedCLIP (medical-specific model via OpenCLIP)
    if OPEN_CLIP_AVAILABLE:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            available['biomedclip'] = True
            print("  BiomedCLIP available (Medical-specific)")
        except Exception as e:
            available['biomedclip'] = False
            print(f"  BiomedCLIP failed: {e}")
    
    # Check OpenAI CLIP
    if TRANSFORMERS_AVAILABLE:
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            available['clip'] = True
            print("  OpenAI CLIP available")
        except:
            available['clip'] = False
    
    # Check DINOv2
    if TIMM_AVAILABLE:
        try:
            model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
            available['dinov2'] = True
            print("  DINOv2 available")
        except:
            available['dinov2'] = False
    
    return available

class BiomedCLIPFoundation(nn.Module):
    """BiomedCLIP foundation model wrapper."""
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        print("  Loading BiomedCLIP foundation model...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        # Get feature dimension from the vision encoder
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.model.encode_image(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Intelligent freezing strategy
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze last 3 transformer layers for medical adaptation
            try:
                for param in self.model.visual.transformer.resblocks[-3:].parameters():
                    param.requires_grad = True
                print("  Unfroze last 3 transformer layers for medical adaptation")
            except:
                print("  Could not unfreeze transformer layers")
        
        # Medical-optimized classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        print(f"  BiomedCLIP loaded: {self.feature_dim} features → {num_classes} classes")
    
    def forward(self, x, return_features=False):
        features = self.model.encode_image(x)
        logits = self.classifier(features)
        if return_features:
            return {'logits': logits, 'features': features}
        return logits

class DINOv2Foundation(nn.Module):
    """DINOv2 foundation model wrapper with input size fix."""
    def __init__(self, num_classes=2, model_name='dinov2_vitb14'):
        super().__init__()
        print(f"Loading DINOv2 foundation model ({model_name})...")
        
        # Try different DINOv2 variants that support 224x224 input
        dinov2_variants = [
            'dinov2_vitb14',
            'vit_base_patch14_dinov2.lvd142m',
            'dinov2_vits14',
            'dinov2_vitl14'
        ]
        
        self.backbone = None
        for variant in dinov2_variants:
            try:
                self.backbone = timm.create_model(variant, pretrained=True, num_classes=0, img_size=224)
                model_name = variant
                print(f"  Successfully loaded {variant}")
                break
            except Exception as e:
                print(f"  Failed to load {variant}: {e}")
                continue
        
        if self.backbone is None:
            raise RuntimeError("Could not load any DINOv2 variant")
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Add input resizing layer as backup (for models without internal resizing)
        self.input_resize = nn.AdaptiveAvgPool2d((224, 224)) if hasattr(self.backbone, 'patch_embed') else None
        
        # Freeze entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last few blocks for fine-tuning
        try:
            if hasattr(self.backbone, 'blocks'):
                for param in self.backbone.blocks[-4:].parameters():
                    param.requires_grad = True
                print("  Unfroze last 4 transformer blocks")
        except:
            print("  Could not unfreeze transformer blocks")
        
        # Medical-optimized classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        print(f"  DINOv2 loaded: {self.feature_dim} features → {num_classes} classes")
    
    def forward(self, x, return_features=False):
        # Ensure input size is 224x224
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        try:
            features = self.backbone(x)
        except Exception as e:
            print(f"  DINOv2 forward failed: {e}")
            batch_size = x.size(0)
            features = torch.zeros(batch_size, self.feature_dim).to(x.device)
        
        logits = self.classifier(features)
        if return_features:
            return {'logits': logits, 'features': features}
        return logits

class CLIPFoundation(nn.Module):
    """OpenAI CLIP foundation model wrapper."""
    def __init__(self, num_classes=2, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        print(f" Loading CLIP foundation model ({model_name})...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.feature_dim = self.model.config.vision_config.hidden_size
        
        # Freeze entire model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last vision encoder layers for adaptation
        for param in self.model.vision_model.encoder.layers[-3:].parameters():
            param.requires_grad = True
        
        # Medical classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        print(f"  CLIP loaded: {self.feature_dim} features → {num_classes} classes")
    
    def forward(self, x, return_features=False):
        vision_outputs = self.model.vision_model(pixel_values=x)
        pooled_output = vision_outputs.pooler_output
        logits = self.classifier(pooled_output)
        if return_features:
            return {'logits': logits, 'features': pooled_output}
        return logits

class FoundationModelTeacher(nn.Module):
    """Heavy teacher ensemble using multiple foundation models."""
    def __init__(self, num_classes=2):
        super().__init__()
        print("Building Foundation Model Teacher Ensemble...")
        self.models = nn.ModuleDict()
        self.available_models = detect_available_foundation_models()
        
        # Add available foundation models to the ensemble
        model_success = {}
        if self.available_models.get('biomedclip', False):
            try:
                self.models['biomedclip'] = BiomedCLIPFoundation(num_classes, freeze_backbone=False)
                model_success['biomedclip'] = True
            except Exception as e:
                print(f"  Failed to load BiomedCLIP: {e}")
                model_success['biomedclip'] = False
        
        if self.available_models.get('dinov2', False):
            try:
                self.models['dinov2'] = DINOv2Foundation(num_classes)
                model_success['dinov2'] = True
            except Exception as e:
                print(f"  Failed to load DINOv2: {e}")
                model_success['dinov2'] = False
        
        if self.available_models.get('clip', False):
            try:
                self.models['clip'] = CLIPFoundation(num_classes)
                model_success['clip'] = True
            except Exception as e:
                print(f"  Failed to load CLIP: {e}")
                model_success['clip'] = False
        
        # If no foundation models loaded, fall back to strong classical models
        if len(self.models) == 0:
            print("  No foundation models available, using classical models")
            self.models['resnet152'] = self._create_resnet152(num_classes)
            self.models['efficientnet_b7'] = self._create_efficientnet_b7(num_classes)
            self.models['vit_large'] = self._create_vit_large(num_classes)
        
        # Ensemble weights (learnable)
        self.num_models = len(self.models)
        self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        print(f"  Teacher ensemble created with {self.num_models} models: {list(self.models.keys())}")
    
    def _create_resnet152(self, num_classes):
        model = models.resnet152(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        return model
    
    def _create_efficientnet_b7(self, num_classes):
        if TIMM_AVAILABLE:
            model = timm.create_model('efficientnet_b7', pretrained=True, num_classes=num_classes)
        else:
            model = models.efficientnet_b7(pretrained=True)
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(2560, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes)
            )
        return model
    
    def _create_vit_large(self, num_classes):
        if TIMM_AVAILABLE:
            model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)
        else:
            # Fallback to ResNet50 if ViT not available
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, num_classes)
        return model
    
    def forward(self, x):
        outputs = []
        successful_models = []
        
        for name, model in self.models.items():
            try:
                output = model(x)
                if isinstance(output, dict):
                    output = output['logits']
                outputs.append(output)
                successful_models.append(name)
            except Exception as e:
                print(f"  Model {name} failed: {str(e)[:100]}...")
                batch_size = x.size(0)
                dummy_output = torch.zeros(batch_size, 2, device=x.device, dtype=x.dtype)
                outputs.append(dummy_output)
        
        if len(successful_models) == 0:
            print("  All models failed! Using dummy ensemble.")
            dummy_ensemble = torch.zeros(x.size(0), 2, device=x.device, dtype=x.dtype)
            return {
                'ensemble_logits': dummy_ensemble,
                'individual_logits': outputs,
                'weights': torch.ones(len(outputs)) / len(outputs),
                'stacked_outputs': torch.stack(outputs, dim=1),
                'successful_models': successful_models
            }
        
        # Stack outputs from all models: shape [batch, num_models, num_classes]
        stacked_outputs = torch.stack(outputs, dim=1)
        
        # Compute weighted ensemble (adjusting for any failed models)
        if len(successful_models) < len(self.models):
            adjusted_weights = self.ensemble_weights.clone()
            for i, name in enumerate(self.models.keys()):
                if name not in successful_models:
                    adjusted_weights[i] = 0.0
            adjusted_weights = F.softmax(adjusted_weights, dim=0)
        else:
            adjusted_weights = F.softmax(self.ensemble_weights, dim=0)
        
        ensemble_logits = torch.sum(stacked_outputs * adjusted_weights.view(1, -1, 1), dim=1)
        
        print(f"  Ensemble using {len(successful_models)}/{len(self.models)} models: {successful_models}")
        
        return {
            'ensemble_logits': ensemble_logits,
            'individual_logits': outputs,
            'weights': adjusted_weights,
            'stacked_outputs': stacked_outputs,
            'successful_models': successful_models
        }

class MedicalEfficientBlock(nn.Module):
    """Medical-optimized efficient block with proper capacity and fixed SE attention."""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Calculate expanded hidden dimensions
        if expand_ratio != 1:
            hidden_dim = int(round(in_channels * expand_ratio))
            self.use_expansion = True
            # Expansion layer
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            hidden_dim = in_channels
            self.use_expansion = False
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 5, stride, 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        # Squeeze-and-Excitation (SE) attention
        se_channels = max(1, hidden_dim // 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_channels, hidden_dim, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Projection to out_channels
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.use_expansion:
            x = self.expand_conv(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # SE attention
        se_weight = self.se(x)
        x = x * se_weight
        
        # Projection
        x = self.project_conv(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x

class LightweightMedicalStudent(nn.Module):
    """Lightweight student network optimized for medical imaging."""
    def __init__(self, num_classes=2, width_multiplier=1.0):
        super().__init__()
        
        # Helper to make layer channels divisible by 8
        def make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            return new_v
        
        # Initial convolutional stem
        input_channel = make_divisible(32 * width_multiplier)
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Configuration for subsequent stages: [expand_ratio, output_channels, num_blocks, stride]
        configs = [
            [1, 24, 2, 1],   # Stage 1: 24 channels
            [2, 32, 2, 2],   # Stage 2: 32 channels (reduced expand_ratio)
            [2, 48, 3, 2],   # Stage 3: 48 channels
            [4, 64, 3, 2],   # Stage 4: 64 channels
            [4, 96, 2, 1],   # Stage 5: 96 channels
            [6, 128, 2, 2]   # Stage 6: 128 channels
        ]
        
        # Build stages using MedicalEfficientBlock
        self.stages = nn.ModuleList()
        for expand_ratio, channels, num_blocks, stride in configs:
            output_channel = make_divisible(channels * width_multiplier)
            stage_layers = []
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                stage_layers.append(MedicalEfficientBlock(input_channel, output_channel, s, expand_ratio))
                input_channel = output_channel
            self.stages.append(nn.Sequential(*stage_layers))
        
        # Final convolution layer before global pooling
        last_channel = make_divisible(256 * width_multiplier)  # reduced final channels for simplicity
        self.final_conv = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Classifier head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
        
        # Print model size (number of parameters)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Lightweight student created: {total_params:,} parameters")
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, extract_features=False):
        x = self.stem(x)
        
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        x = self.final_conv(x)
        global_features = self.global_pool(x).flatten(1)
        logits = self.classifier(global_features)
        
        if extract_features:
            return {
                'logits': logits,
                'features': global_features,
                'stage_features': features
            }
        return logits