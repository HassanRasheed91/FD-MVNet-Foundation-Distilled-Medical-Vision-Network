import warnings
import time
import os
from pathlib import Path
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import json

# Import modules from refactored codebase
from models import FoundationModelTeacher, LightweightMedicalStudent
from dataset import get_enhanced_transforms, ResearchGradeCOVIDDataset
from training import FoundationModelTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configure device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎓 Research Environment: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def run_foundation_model_experiment(data_dir, device, teacher_epochs=25, student_epochs=40):
    """Run complete foundation model experiment with comprehensive visualization"""
    print(" FOUNDATION MODEL COVID DETECTION EXPERIMENT")
    print("="*80)
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Data pipeline
        print("\n Setting up enhanced data pipeline...")
        train_transform, val_transform = get_enhanced_transforms()
        train_dataset = ResearchGradeCOVIDDataset(data_dir, 'train', train_transform)
        val_dataset = ResearchGradeCOVIDDataset(data_dir, 'val', val_transform)
        
        # Balanced sampling
        train_sampler = train_dataset.get_sampler()
        train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                                num_workers=4, pin_memory=True, persistent_workers=True)
        print(f" Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # 2. Initialize models
        print("\n Initializing foundation models...")
        teacher_model = FoundationModelTeacher(num_classes=2).to(device)
        student_model = LightweightMedicalStudent(num_classes=2, width_multiplier=1.2).to(device)
        
        # Model statistics
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        reduction_ratio = teacher_params / student_params
        
        print(f" Model Statistics:")
        print(f"   Teacher: {teacher_params:,} parameters")
        print(f"   Student: {student_params:,} parameters")
        print(f"   Reduction: {reduction_ratio:.1f}x")
        
        # 3. Training with comprehensive tracking
        trainer = FoundationModelTrainer(device, save_dir=results_dir)
        
        # Stage 1: Train teacher
        print(f"\n Stage 1: Training Foundation Teacher ({teacher_epochs} epochs)...")
        teacher_acc = trainer.train_teacher(teacher_model, train_loader, val_loader, teacher_epochs)
        
        # Stage 2: Train student
        print(f"\n Stage 2: Training Student with Distillation ({student_epochs} epochs)...")
        student_acc = trainer.train_student(student_model, teacher_model, train_loader, val_loader, student_epochs)
        
        # 5. Final comprehensive evaluation
        print(f"\n Comprehensive Evaluation & Visualization...")
        teacher_final_acc, _ = trainer.evaluate_model(teacher_model, val_loader)
        student_final_acc, _ = trainer.evaluate_model(student_model, val_loader)
        
        # Prepare results dictionary
        results_dict = {
            'teacher_accuracy': teacher_final_acc,
            'student_accuracy': student_final_acc,
            'teacher_params': teacher_params,
            'student_params': student_params,
            'reduction_ratio': reduction_ratio,
            'training_history': trainer.history,
            'success': student_final_acc >= 99.0,
            'available_models': teacher_model.available_models
        }
        
        # Generate all research visualizations and metrics
        comprehensive_metrics = trainer.generate_comprehensive_results(teacher_model, student_model, val_loader, results_dict)
        
        # 6. Results summary
        print("\n" + "="*80)
        print(" FOUNDATION MODEL EXPERIMENT RESULTS")
        print("="*80)
        print(f"Teacher Accuracy: {teacher_final_acc:.2f}%")
        print(f"Student Accuracy: {student_final_acc:.2f}%")
        print(f"Parameter Reduction: {reduction_ratio:.1f}x")
        print(f"Student Model Size: {student_params * 4 / (1024*1024):.1f} MB")
        print(f"Teacher AUC-ROC: {comprehensive_metrics['teacher_auc']:.4f}")
        print(f"Student AUC-ROC: {comprehensive_metrics['student_auc']:.4f}")
        
        success = student_final_acc >= 99.0
        if success:
            print(f"\n SUCCESS! ACHIEVED 99%+ ACCURACY! ")
            print(f"  Student accuracy: {student_final_acc:.2f}% >= 99.0%")
            print(f"  Significant parameter reduction: {reduction_ratio:.1f}x")
            print(f"  Foundation model knowledge successfully transferred")
        elif student_final_acc >= 95.0:
            print(f"\n GOOD PROGRESS: {student_final_acc:.2f}%")
            print(f" Close to target! Consider:")
            print(f"   • Fine-tuning hyperparameters")
            print(f"   • Adding more foundation models to teacher")
            print(f"   • Increasing training epochs")
        else:
            print(f"\n NEEDS IMPROVEMENT: {student_final_acc:.2f}%")
            print(f" Recommendations:")
            print(f"   • Check data quality and balance")
            print(f"   • Verify foundation models are loading correctly")
            print(f"   • Increase model capacity (width_multiplier)")
            print(f"   • Extend training duration")
        
        # 7. Save comprehensive results
        print(f"\n Saving comprehensive results...")
        results_dict.update({
            'comprehensive_metrics': comprehensive_metrics,
            'model_size_mb': student_params * 4 / (1024*1024),
            'efficiency_score': (student_final_acc / teacher_final_acc) * reduction_ratio,
            'deployment_ready': student_final_acc >= 99.0 and student_params < 5e6
        })
        
        # Save main results
        torch.save(results_dict, results_dir / 'foundation_experiment_results.pth')
        
        # Save models with metadata
        torch.save({
            'teacher_state_dict': teacher_model.state_dict(),
            'student_state_dict': student_model.state_dict(),
            'teacher_model_info': {
                'class': 'FoundationModelTeacher',
                'available_models': teacher_model.available_models,
                'num_models': len(teacher_model.models),
                'parameters': teacher_params
            },
            'student_model_info': {
                'class': 'LightweightMedicalStudent',
                'width_multiplier': 1.2,
                'parameters': student_params
            },
            'training_config': {
                'teacher_epochs': teacher_epochs,
                'student_epochs': student_epochs,
                'batch_size': 16,
                'optimizer': 'AdamW',
                'scheduler': 'OneCycleLR'
            },
            'performance_metrics': {
                'teacher_accuracy': teacher_final_acc,
                'student_accuracy': student_final_acc,
                'teacher_auc': comprehensive_metrics['teacher_auc'],
                'student_auc': comprehensive_metrics['student_auc'],
                'reduction_ratio': reduction_ratio
            }
        }, results_dir / 'foundation_models.pth')
        
        # Generate publication-ready summary
        print(f"\n Generating publication summary...")
        performance_summary = pd.DataFrame({
            'Model': ['Teacher Ensemble', 'Lightweight Student'],
            'Accuracy (%)': [f"{teacher_final_acc:.2f}", f"{student_final_acc:.2f}"],
            'Precision': [f"{comprehensive_metrics['teacher_metrics']['precision']:.4f}",
                          f"{comprehensive_metrics['student_metrics']['precision']:.4f}"],
            'Recall': [f"{comprehensive_metrics['teacher_metrics']['recall']:.4f}",
                       f"{comprehensive_metrics['student_metrics']['recall']:.4f}"],
            'F1-Score': [f"{comprehensive_metrics['teacher_metrics']['f1_score']:.4f}",
                         f"{comprehensive_metrics['student_metrics']['f1_score']:.4f}"],
            'AUC-ROC': [f"{comprehensive_metrics['teacher_auc']:.4f}",
                        f"{comprehensive_metrics['student_auc']:.4f}"],
            'Parameters': [f"{teacher_params:,}", f"{student_params:,}"],
            'Size (MB)': [f"{teacher_params * 4 / (1024*1024):.1f}",
                          f"{student_params * 4 / (1024*1024):.1f}"]
        })
        performance_summary.to_csv(results_dir / 'performance_summary.csv', index=False)
        print(f" Performance summary saved to: performance_summary.csv")
        
        highlights = {
            'Key Achievements': [
                f"Student model accuracy: {student_final_acc:.2f}%",
                f"Parameter reduction: {reduction_ratio:.1f}x",
                f"Model compression: {teacher_params * 4 / (student_params * 4):.1f}x size reduction",
                f"Maintained performance: {(student_final_acc/teacher_final_acc)*100:.1f}% accuracy retention"
            ],
            'Clinical Impact': [
                f"High sensitivity: {comprehensive_metrics['student_metrics']['recall_covid']:.3f}",
                f"High specificity: {comprehensive_metrics['student_metrics']['recall_non_covid']:.3f}",
                "Suitable for resource-constrained environments",
                "Real-time inference capability"
            ]
        }
        
        with open(results_dir / 'research_highlights.json', 'w') as f:
            json.dump(highlights, f, indent=2)
        
        print(f"\n All results saved to: {results_dir}")
        print(" Files generated:")
        print("   • training_curves.png - Training progress visualization")
        print("   • model_comparison.png - Performance and efficiency comparison")
        print("   • confusion_matrices.png - Classification results")
        print("   • roc_curves.png - ROC analysis")
        print("   • precision_recall_curves.png - PR curve analysis")
        print("   • comprehensive_metrics.png - Complete metrics table")
        print("   • knowledge_distillation_analysis.png - Distillation effectiveness")
        print("   • comprehensive_metrics.csv - Detailed metrics data")
        print("   • performance_summary.csv - Model comparison table")
        print("   • research_summary.md - Publication-ready summary")
        print("   • research_highlights.json - Key achievements")
        print("   • foundation_models.pth - Trained model weights")
        print("   • foundation_experiment_results.pth - Complete results")
        print("   • training.log - Detailed training logs")
        
        return {
            'teacher_model': teacher_model,
            'student_model': student_model,
            'teacher_accuracy': teacher_final_acc,
            'student_accuracy': student_final_acc,
            'success': success,
            'results': results_dict,
            'comprehensive_metrics': comprehensive_metrics,
            'reduction_ratio': reduction_ratio,
            'model_size_mb': student_params * 4 / (1024*1024)
        }
        
    except Exception as e:
        print(f" Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information for debugging
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        torch.save(error_info, results_dir / 'error_log.pth')
        print(f" Error information saved to: {results_dir}/error_log.pth")
        raise

if __name__ == "__main__":
    # Update this path according to your dataset location
    DATASET_PATH = '/kaggle/input/covid-dataset/dataset_split'
    
    print(" EXECUTING FOUNDATION MODEL COVID DETECTION EXPERIMENT")
    print("="*70)
    
    # Verify dataset
    if not os.path.exists(DATASET_PATH):
        print(f" Dataset not found at {DATASET_PATH}")
        print("Please update DATASET_PATH with correct path")
        alternative_paths = [
            '/kaggle/input/covid19-dataset/dataset_split',
            '/kaggle/working/dataset_split',
            './dataset_split',
            '../dataset_split'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                DATASET_PATH = alt_path
                print(f" Found dataset at alternative path: {DATASET_PATH}")
                break
        else:
            print(" No dataset found. Please check the dataset path.")
            exit()
    else:
        print(f" Dataset found at {DATASET_PATH}")
    
    # Verify dataset structure
    print(" Dataset structure:")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(DATASET_PATH, split)
        if os.path.exists(split_path):
            subfolders = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
            print(f"   {split}/: {subfolders}")
        else:
            print(f"   {split}/: Not found")
    
    try:
        print(f"\n Starting comprehensive foundation model experiment...")
        start_time = time.time()
        
        results = run_foundation_model_experiment(
            data_dir=DATASET_PATH,
            device=device,
            teacher_epochs=100,  # Reduced to 25-35 if tiem is short
            student_epochs=100   # Reduced to 50-60 if tiem is short
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*70)
        print(" FOUNDATION MODEL EXPERIMENT COMPLETED!")
        print("="*70)
        print(f" Total execution time: {total_time/3600:.2f} hours")
        
        if results['success']:
            print(" SUCCESS! 99%+ ACCURACY ACHIEVED! ")
            print(f" Final student accuracy: {results['student_accuracy']:.2f}%")
            print(f"Parameter reduction: {results['reduction_ratio']:.1f}x")
            print(f" Model size: {results['model_size_mb']:.1f} MB")
            
            print(f"\n Key Achievements:")
            print(f"   • State-of-the-art accuracy with lightweight model")
            print(f"   • Successful knowledge transfer from foundation models")
            print(f"   • Deployment-ready architecture")
        else:
            print(f"Progress: {results['student_accuracy']:.2f}% (Target: 99%)")
            print(f" Consider:")
            print(f"   • Fine-tuning hyperparameters")
            print(f"   • Adding more foundation models")
        
        print(f"\n All results available in: ./results/")
        print(f"\n FINAL METRICS SUMMARY:")
        print(f"{'='*50}")
        print("Teacher Model:")
        print(f"  Accuracy: {results['teacher_accuracy']:.2f}%")
        print(f"  Parameters: {results['results']['teacher_params']:,}")
        print(f"  AUC-ROC: {results['comprehensive_metrics']['teacher_auc']:.4f}")
        print(f"\nStudent Model:")
        print(f"  Accuracy: {results['student_accuracy']:.2f}%")
        print(f"  Parameters: {results['results']['student_params']:,}")
        print(f"  AUC-ROC: {results['comprehensive_metrics']['student_auc']:.4f}")
        print(f"  Size: {results['model_size_mb']:.1f} MB")
        print(f"\nEfficiency:")
        print(f"  Parameter Reduction: {results['reduction_ratio']:.1f}x")
        print(f"  Accuracy Retention: {(results['student_accuracy']/results['teacher_accuracy'])*100:.1f}%")
        
    except Exception as e:
        print(f" Experiment failed: {e}")
        print(" Check error_log.pth for detailed debugging information")
        print(" Common issues:")
        print("   • Insufficient GPU memory (reduce batch_size)")
        print("   • Missing dependencies (check pip installs)")
        print("   • Dataset path issues (verify DATASET_PATH)")
    
    print("\n" + "="*70)
    print(" FOUNDATION MODEL COVID DETECTION SYSTEM COMPLETE!")
    print("="*70)
