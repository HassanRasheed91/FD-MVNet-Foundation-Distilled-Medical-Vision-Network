import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from visualization import ResearchVisualizationEngine

class AdvancedDistillationLoss(nn.Module):
    """Advanced knowledge distillation loss for medical AI."""
    def __init__(self, temperature=3.0, alpha=0.8, feature_weight=0.2):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_outputs, teacher_outputs, targets):
        # Extract logits from student and teacher outputs
        student_logits = student_outputs['logits'] if isinstance(student_outputs, dict) else student_outputs
        teacher_logits = teacher_outputs['ensemble_logits']
        
        # Hard target loss (ground truth labels)
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Soft target loss (distillation from teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Ensemble consistency loss (ensure student aligns with each teacher model)
        ensemble_loss = 0.0
        if 'individual_logits' in teacher_outputs:
            individual_logits = teacher_outputs['individual_logits']
            for teacher_logit in individual_logits:
                teacher_soft_i = F.softmax(teacher_logit / self.temperature, dim=1)
                ensemble_loss += self.kl_loss(student_soft, teacher_soft_i)
            ensemble_loss = ensemble_loss / len(individual_logits) * 0.1
        
        # Total loss is a weighted sum of hard and soft losses (plus ensemble consistency)
        total_loss = (
            (1 - self.alpha) * hard_loss +
            self.alpha * soft_loss +
            ensemble_loss
        )
        
        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss,
            'ensemble_loss': ensemble_loss
        }

class FoundationModelTrainer:
    """
    Trainer class that handles training of the teacher ensemble and student model
    """
    def __init__(self, device, save_dir='./results'):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set up logging to file and stdout
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
        self.history = {'teacher': {}, 'student': {}}
        
        # Initialize the visualization engine for plotting results
        self.viz_engine = ResearchVisualizationEngine(save_dir=save_dir)
    
    def train_teacher(self, teacher_model, train_loader, val_loader, epochs=25):
        """Stage 1: Train the foundation model teacher ensemble."""
        logging.info(" STAGE 1: Training Foundation Model Teacher Ensemble")
        logging.info("="*60)
        
        teacher_model.train()
        
        # Set up optimizer with different LRs for different components
        param_groups = []
        for name, model in teacher_model.models.items():
            if 'biomedclip' in name or 'clip' in name or 'dinov2' in name:
                # Lower learning rate for foundation models
                param_groups.append({'params': model.parameters(), 'lr': 1e-5})
            else:
                # Higher learning rate for classical models
                param_groups.append({'params': model.parameters(), 'lr': 1e-4})
        
        # Include ensemble weight parameters with a separate learning rate
        param_groups.append({'params': [teacher_model.ensemble_weights], 'lr': 1e-3})
        
        optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            teacher_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = teacher_model(data)
                loss = criterion(outputs['ensemble_logits'], target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                pred = outputs['ensemble_logits'].argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
                
                if batch_idx % 50 == 0:
                    logging.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            scheduler.step()
            
            # Evaluate on validation set
            val_acc, val_loss = self.evaluate_model(teacher_model, val_loader)
            train_acc = 100. * train_correct / train_total
            
            # Record history
            if 'train_loss' not in self.history['teacher']:
                for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
                    self.history['teacher'][key] = []
            
            self.history['teacher']['train_loss'].append(train_loss / len(train_loader))
            self.history['teacher']['train_acc'].append(train_acc)
            self.history['teacher']['val_loss'].append(val_loss)
            self.history['teacher']['val_acc'].append(val_acc)
            
            # Save model if improved
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(teacher_model.state_dict(), self.save_dir / 'teacher_best.pth')
                logging.info(f" New best teacher accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
                         f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # Early stopping if no improvement
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model weights before returning
        teacher_model.load_state_dict(torch.load(self.save_dir / 'teacher_best.pth'))
        logging.info(f" Teacher training completed. Best accuracy: {best_acc:.2f}%")
        
        return best_acc
    
    def train_student(self, student_model, teacher_model, train_loader, val_loader, epochs=40):
        """Stage 2: Train the student model with knowledge distillation."""
        logging.info(" STAGE 2: Training Student with Knowledge Distillation")
        logging.info("="*60)
        
        # Freeze teacher model parameters
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        student_model.train()
        
        optimizer = optim.AdamW(student_model.parameters(), lr=2e-4, weight_decay=0.02)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs,
            steps_per_epoch=len(train_loader), pct_start=0.3
        )
        
        distillation_loss = AdvancedDistillationLoss(temperature=3.0, alpha=0.8)
        
        best_acc = 0
        patience = 30
        patience_counter = 0
        
        for epoch in range(epochs):
            student_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            loss_components = {'total': 0, 'hard': 0, 'soft': 0, 'ensemble': 0}
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Get teacher ensemble outputs (no grad)
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                
                # Get student outputs (with feature extraction for loss)
                student_outputs = student_model(data, extract_features=True)
                
                # Compute distillation loss
                loss_dict = distillation_loss(student_outputs, teacher_outputs, target)
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Track training loss and accuracy
                train_loss += loss.item()
                pred = student_outputs['logits'].argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
                
                # Accumulate loss components for logging
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[key.replace('_loss', '')] += value.item()
                
                if batch_idx % 50 == 0:
                    logging.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # Evaluate on validation set
            val_acc, val_loss = self.evaluate_model(student_model, val_loader)
            train_acc = 100. * train_correct / train_total
            
            # Record history
            if 'train_loss' not in self.history['student']:
                for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
                    self.history['student'][key] = []
            
            self.history['student']['train_loss'].append(train_loss / len(train_loader))
            self.history['student']['train_acc'].append(train_acc)
            self.history['student']['val_loss'].append(val_loss)
            self.history['student']['val_acc'].append(val_acc)
            
            # Save model if improved
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(student_model.state_dict(), self.save_dir / 'student_best.pth')
                logging.info(f" New best student accuracy: {val_acc:.2f}%")
                
                if val_acc >= 99.0:
                    logging.info(f" TARGET ACHIEVED! {val_acc:.2f}% >= 99.0%")
            else:
                patience_counter += 1
            
            # Periodic logging of distillation loss components
            if epoch % 5 == 0:
                logging.info(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
                logging.info(f"  Loss components - Hard: {loss_components['hard']/len(train_loader):.4f}, "
                           f"Soft: {loss_components['soft']/len(train_loader):.4f}")
            
            # Early stopping if no improvement for a while (after some epochs)
            if patience_counter >= patience and epoch > 20:
                logging.info(f" Early stopping at epoch {epoch+1}")
                break
        
        # Load best model weights
        student_model.load_state_dict(torch.load(self.save_dir / 'student_best.pth'))
        logging.info(f"🎓 Student training completed. Best accuracy: {best_acc:.2f}%")
        
        return best_acc
    
    def evaluate_model(self, model, data_loader):
        """Evaluate model performance on a given dataset loader."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = model(data)
                if isinstance(outputs, dict):
                    if 'ensemble_logits' in outputs:
                        logits = outputs['ensemble_logits']
                    else:
                        logits = outputs['logits']
                else:
                    logits = outputs
                
                loss = criterion(logits, target)
                total_loss += loss.item()
                
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(data_loader)
        
        return accuracy, avg_loss
    
    def generate_comprehensive_results(self, teacher_model, student_model, test_loader, results_dict):
        """Generate comprehensive research results with all visualizations."""
        logging.info(" Generating comprehensive research visualizations...")
        
        # 1. Training curves
        self.viz_engine.plot_training_curves(self.history, "Foundation Model COVID Detection Training")
        
        # 2. Model comparison
        self.viz_engine.plot_model_comparison(results_dict)
        
        # 3. Confusion matrices and get prediction data
        teacher_data, student_data = self.viz_engine.plot_confusion_matrices(
            teacher_model, student_model, test_loader, self.device
        )
        
        # 4. ROC curves
        teacher_auc, student_auc = self.viz_engine.plot_roc_curves(teacher_data, student_data)
        
        # 5. Precision-Recall curves
        teacher_ap, student_ap = self.viz_engine.plot_precision_recall_curves(teacher_data, student_data)
        
        # 6. Comprehensive metrics report (as table and CSV)
        teacher_metrics, student_metrics = self.viz_engine.generate_comprehensive_metrics_report(
            teacher_data, student_data, results_dict
        )
        
        # 7. Knowledge distillation analysis plots
        self.viz_engine.plot_knowledge_distillation_analysis(self.history)
        
        # 8. Generate a research summary markdown file
        self.generate_research_summary(results_dict, teacher_metrics, student_metrics, teacher_auc, student_auc)
        
        logging.info(" All research visualizations generated successfully!")
        
        return {
            'teacher_metrics': teacher_metrics,
            'student_metrics': student_metrics,
            'teacher_auc': teacher_auc,
            'student_auc': student_auc,
            'teacher_ap': teacher_ap,
            'student_ap': student_ap
        }
    
    def generate_research_summary(self, results_dict, teacher_metrics, student_metrics, teacher_auc, student_auc):
        """Generate a comprehensive research summary in Markdown format."""
        summary_path = self.save_dir / 'research_summary.md'
        
        with open(summary_path, 'w') as f:
            f.write("# Foundation Model-Enhanced COVID Detection System\n")
            f.write("## Research Summary & Results\n\n")
            
            f.write("###  Key Achievements\n")
            f.write(f"- **Student Model Accuracy**: {results_dict.get('student_accuracy', 0):.2f}%\n")
            f.write(f"- **Teacher Model Accuracy**: {results_dict.get('teacher_accuracy', 0):.2f}%\n")
            f.write(f"- **Parameter Reduction**: {results_dict.get('reduction_ratio', 0):.1f}x\n")
            f.write(f"- **Model Size**: {results_dict.get('student_params', 0) * 4 / (1024*1024):.1f} MB\n\n")
            
            f.write("###  Detailed Performance Metrics\n\n")
            f.write("#### Teacher Ensemble Model\n")
            f.write(f"- Accuracy: {teacher_metrics['accuracy']:.4f}\n")
            f.write(f"- Precision: {teacher_metrics['precision']:.4f}\n")
            f.write(f"- Recall: {teacher_metrics['recall']:.4f}\n")
            f.write(f"- F1-Score: {teacher_metrics['f1_score']:.4f}\n")
            f.write(f"- AUC-ROC: {teacher_auc:.4f}\n\n")
            
            f.write("#### Lightweight Student Model\n")
            f.write(f"- Accuracy: {student_metrics['accuracy']:.4f}\n")
            f.write(f"- Precision: {student_metrics['precision']:.4f}\n")
            f.write(f"- Recall: {student_metrics['recall']:.4f}\n")
            f.write(f"- F1-Score: {student_metrics['f1_score']:.4f}\n")
            f.write(f"- AUC-ROC: {student_auc:.4f}\n\n")
            
            f.write("###  Generated Files\n")
            f.write("- `training_curves.png`: Training progress visualization\n")
            f.write("- `model_comparison.png`: Performance and efficiency comparison\n")
            f.write("- `confusion_matrices.png`: Detailed classification results\n")
            f.write("- `roc_curves.png`: ROC analysis for both models\n")
            f.write("- `precision_recall_curves.png`: PR curve analysis\n")
            f.write("- `comprehensive_metrics.png`: Complete metrics table\n")
            f.write("- `knowledge_distillation_analysis.png`: Distillation effectiveness\n")
            f.write("- `comprehensive_metrics.csv`: Metrics data for further analysis\n")
            f.write("- `foundation_models.pth`: Trained model weights\n")
            f.write("- `training.log`: Complete training logs\n\n")
        
        logging.info(f" Research summary saved to: {summary_path}")
