import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score

class ResearchVisualizationEngine:
    """Comprehensive visualization engine for research paper results."""
    def __init__(self, save_dir='./results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set a clean, publication-ready style for plots
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Configure Matplotlib global parameters for high-quality figures
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def plot_training_curves(self, history, title="Foundation Model Training Curves"):
        """Plot training and validation loss/accuracy curves for teacher and student."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # Teacher Training Curves
        if 'teacher' in history and history['teacher']:
            teacher_history = history['teacher']
            epochs_teacher = range(1, len(teacher_history['train_loss']) + 1)
            
            # Teacher Loss
            axes[0, 0].plot(epochs_teacher, teacher_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            axes[0, 0].plot(epochs_teacher, teacher_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            axes[0, 0].set_title('Teacher Model - Loss Curves', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Teacher Accuracy
            axes[0, 1].plot(epochs_teacher, teacher_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
            axes[0, 1].plot(epochs_teacher, teacher_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
            axes[0, 1].axhline(y=99, color='g', linestyle='--', alpha=0.7, label='Target (99%)')
            axes[0, 1].set_title('Teacher Model - Accuracy Curves', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Student Training Curves
        if 'student' in history and history['student']:
            student_history = history['student']
            epochs_student = range(1, len(student_history['train_loss']) + 1)
            
            # Student Loss
            axes[1, 0].plot(epochs_student, student_history['train_loss'], color='purple', label='Train Loss', linewidth=2)
            axes[1, 0].plot(epochs_student, student_history['val_loss'], color='orange', label='Val Loss', linewidth=2)
            axes[1, 0].set_title('Student Model - Loss Curves', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Student Accuracy
            axes[1, 1].plot(epochs_student, student_history['train_acc'], color='purple', label='Train Acc', linewidth=2)
            axes[1, 1].plot(epochs_student, student_history['val_acc'], color='orange', label='Val Acc', linewidth=2)
            axes[1, 1].axhline(y=99, color='g', linestyle='--', alpha=0.7, label='Target (99%)')
            axes[1, 1].set_title('Student Model - Accuracy Curves', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results_dict):
        """Create a side-by-side comparison of teacher vs student performance and size."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Foundation Model vs Student: Performance & Efficiency', fontsize=16, fontweight='bold')
        
        models = ['Teacher Ensemble', 'Lightweight Student']
        accuracies = [results_dict.get('teacher_accuracy', 0), results_dict.get('student_accuracy', 0)]
        params = [results_dict.get('teacher_params', 0), results_dict.get('student_params', 0)]
        
        # Accuracy Comparison
        bars1 = axes[0].bar(models, accuracies, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        axes[0].axhline(y=99, color='red', linestyle='--', alpha=0.7, label='Target (99%)')
        axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_ylim(85, 100)
        
        # Add accuracy labels on bars
        for bar, acc in zip(bars1, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Parameter Comparison (in millions, log scale)
        params_millions = [p / 1e6 for p in params]
        bars2 = axes[1].bar(models, params_millions, color=['#2ca02c', '#d62728'], alpha=0.8)
        axes[1].set_title('Model Size Comparison', fontweight='bold')
        axes[1].set_ylabel('Parameters (Millions)')
        axes[1].set_yscale('log')
        
        # Add parameter labels on bars
        for bar, param in zip(bars2, params_millions):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                         f'{param:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # Efficiency Metrics (Parameter reduction and accuracy retention)
        if results_dict.get('reduction_ratio', 0) > 0:
            reduction_ratio = results_dict['reduction_ratio']
            accuracy_retention = (results_dict.get('student_accuracy', 0) / results_dict.get('teacher_accuracy', 1)) * 100
            
            metrics = ['Parameter Reduction', 'Accuracy Retention']
            values = [reduction_ratio, accuracy_retention]
            colors = ['#9467bd', '#8c564b']
            
            bars3 = axes[2].bar(metrics, values, color=colors, alpha=0.8)
            axes[2].set_title('Efficiency Metrics', fontweight='bold')
            axes[2].set_ylabel('Ratio/Percentage')
            
            # Add value labels for efficiency metrics
            for bar, val, metric in zip(bars3, values, metrics):
                if metric == 'Parameter Reduction':
                    label = f'{val:.1f}x'
                else:
                    label = f'{val:.1f}%'
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                             label, ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, teacher_model, student_model, test_loader, device):
        """Generate confusion matrix plots for teacher and student models."""
        def get_predictions(model, data_loader):
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for data, target in data_loader:
                    data, target = data.to(device), target.to(device)
                    
                    outputs = model(data)
                    if isinstance(outputs, dict):
                        if 'ensemble_logits' in outputs:
                            logits = outputs['ensemble_logits']
                        else:
                            logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    probs = F.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(target.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            return np.array(all_preds), np.array(all_labels), np.array(all_probs)
        
        # Get predictions for teacher and student
        teacher_preds, teacher_labels, teacher_probs = get_predictions(teacher_model, test_loader)
        student_preds, student_labels, student_probs = get_predictions(student_model, test_loader)
        
        # Plot confusion matrices side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Confusion Matrices: Foundation Model vs Student', fontsize=16, fontweight='bold')
        
        class_names = ['Non-COVID', 'COVID']
        
        # Teacher confusion matrix
        teacher_cm = confusion_matrix(teacher_labels, teacher_preds)
        sns.heatmap(teacher_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Teacher Ensemble Model', fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Student confusion matrix
        student_cm = confusion_matrix(student_labels, student_preds)
        sns.heatmap(student_cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Lightweight Student Model', fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return (teacher_preds, teacher_labels, teacher_probs), (student_preds, student_labels, student_probs)
    
    def plot_roc_curves(self, teacher_data, student_data):
        """Plot ROC curves for teacher and student models."""
        teacher_preds, teacher_labels, teacher_probs = teacher_data
        student_preds, student_labels, student_probs = student_data
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('ROC Curves: Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Teacher ROC curve
        teacher_fpr, teacher_tpr, _ = roc_curve(teacher_labels, teacher_probs[:, 1])
        teacher_auc = roc_auc_score(teacher_labels, teacher_probs[:, 1])
        
        axes[0].plot(teacher_fpr, teacher_tpr, 'b-', linewidth=2, label=f'Teacher AUC = {teacher_auc:.4f}')
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].set_title('Teacher Ensemble Model', fontweight='bold')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Student ROC curve
        student_fpr, student_tpr, _ = roc_curve(student_labels, student_probs[:, 1])
        student_auc = roc_auc_score(student_labels, student_probs[:, 1])
        
        axes[1].plot(student_fpr, student_tpr, 'orange', linewidth=2, label=f'Student AUC = {student_auc:.4f}')
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1].set_title('Lightweight Student Model', fontweight='bold')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return teacher_auc, student_auc
    
    def plot_precision_recall_curves(self, teacher_data, student_data):
        """Plot Precision-Recall curves for teacher and student models."""
        teacher_preds, teacher_labels, teacher_probs = teacher_data
        student_preds, student_labels, student_probs = student_data
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Precision-Recall Curves: Medical AI Performance', fontsize=16, fontweight='bold')
        
        # Teacher Precision-Recall curve
        teacher_precision, teacher_recall, _ = precision_recall_curve(teacher_labels, teacher_probs[:, 1])
        teacher_ap = average_precision_score(teacher_labels, teacher_probs[:, 1])
        
        axes[0].plot(teacher_recall, teacher_precision, 'b-', linewidth=2, label=f'Teacher AP = {teacher_ap:.4f}')
        axes[0].set_title('Teacher Ensemble Model', fontweight='bold')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Student Precision-Recall curve
        student_precision, student_recall, _ = precision_recall_curve(student_labels, student_probs[:, 1])
        student_ap = average_precision_score(student_labels, student_probs[:, 1])
        
        axes[1].plot(student_recall, student_precision, 'orange', linewidth=2, label=f'Student AP = {student_ap:.4f}')
        axes[1].set_title('Lightweight Student Model', fontweight='bold')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return teacher_ap, student_ap
    
    def generate_comprehensive_metrics_report(self, teacher_data, student_data, results_dict):
        """Generate and display a comprehensive metrics table for both models."""
        teacher_preds, teacher_labels, teacher_probs = teacher_data
        student_preds, student_labels, student_probs = student_data
        
        # Calculate metrics for each model
        def calculate_metrics(y_true, y_pred, y_probs):
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted'),
                'auc_roc': roc_auc_score(y_true, y_probs[:, 1]),
                'avg_precision': average_precision_score(y_true, y_probs[:, 1])
            }
            
            # Per-class metrics (0 = Non-COVID, 1 = COVID)
            precision_per_class = precision_score(y_true, y_pred, average=None)
            recall_per_class = recall_score(y_true, y_pred, average=None)
            f1_per_class = f1_score(y_true, y_pred, average=None)
            
            metrics['precision_non_covid'] = precision_per_class[0]
            metrics['precision_covid'] = precision_per_class[1]
            metrics['recall_non_covid'] = recall_per_class[0]
            metrics['recall_covid'] = recall_per_class[1]
            metrics['f1_non_covid'] = f1_per_class[0]
            metrics['f1_covid'] = f1_per_class[1]
            
            return metrics
        
        teacher_metrics = calculate_metrics(teacher_labels, teacher_preds, teacher_probs)
        student_metrics = calculate_metrics(student_labels, student_preds, student_probs)
        
        # Prepare a matplotlib table of metrics
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        metrics_names = [
            'Overall Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score',
            'AUC-ROC', 'Average Precision', 'Non-COVID Precision', 'COVID Precision',
            'Non-COVID Recall', 'COVID Recall', 'Non-COVID F1', 'COVID F1'
        ]
        
        teacher_values = [
            f"{teacher_metrics['accuracy']:.4f}",
            f"{teacher_metrics['precision']:.4f}",
            f"{teacher_metrics['recall']:.4f}",
            f"{teacher_metrics['f1_score']:.4f}",
            f"{teacher_metrics['auc_roc']:.4f}",
            f"{teacher_metrics['avg_precision']:.4f}",
            f"{teacher_metrics['precision_non_covid']:.4f}",
            f"{teacher_metrics['precision_covid']:.4f}",
            f"{teacher_metrics['recall_non_covid']:.4f}",
            f"{teacher_metrics['recall_covid']:.4f}",
            f"{teacher_metrics['f1_non_covid']:.4f}",
            f"{teacher_metrics['f1_covid']:.4f}"
        ]
        
        student_values = [
            f"{student_metrics['accuracy']:.4f}",
            f"{student_metrics['precision']:.4f}",
            f"{student_metrics['recall']:.4f}",
            f"{student_metrics['f1_score']:.4f}",
            f"{student_metrics['auc_roc']:.4f}",
            f"{student_metrics['avg_precision']:.4f}",
            f"{student_metrics['precision_non_covid']:.4f}",
            f"{student_metrics['precision_covid']:.4f}",
            f"{student_metrics['recall_non_covid']:.4f}",
            f"{student_metrics['recall_covid']:.4f}",
            f"{student_metrics['f1_non_covid']:.4f}",
            f"{student_metrics['f1_covid']:.4f}"
        ]
        
        table_data = []
        for i, metric in enumerate(metrics_names):
            table_data.append([metric, teacher_values[i], student_values[i]])
        
        table = ax.table(cellText=table_data,
                         colLabels=['Metric', 'Teacher Ensemble', 'Lightweight Student'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Highlight table header
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight accuracy rows if > 99%
        for i in range(1, len(metrics_names) + 1):
            if 'Accuracy' in metrics_names[i-1]:
                teacher_acc = float(teacher_values[i-1])
                student_acc = float(student_values[i-1])
                if teacher_acc > 0.99:
                    table[(i, 1)].set_facecolor('#E8F5E8')
                if student_acc > 0.99:
                    table[(i, 2)].set_facecolor('#E8F5E8')
        
        plt.title('Comprehensive Performance Metrics: Foundation Model vs Student',
                  fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.save_dir / 'comprehensive_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame({
            'Metric': metrics_names,
            'Teacher_Ensemble': teacher_values,
            'Lightweight_Student': student_values
        })
        metrics_df.to_csv(self.save_dir / 'comprehensive_metrics.csv', index=False)
        
        return teacher_metrics, student_metrics
    
    def plot_knowledge_distillation_analysis(self, history):
        """Analyze and visualize the knowledge distillation process over training."""
        if 'student' not in history or not history['student']:
            print("No student training history available for distillation analysis")
            return
        
        student_history = history['student']
        epochs = range(1, len(student_history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Knowledge Distillation Analysis', fontsize=18, fontweight='bold')
        
        # Training loss curve
        axes[0, 0].plot(epochs, student_history['train_loss'], 'purple', linewidth=2, label='Total Loss')
        axes[0, 0].set_title('Student Training Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy improvement curve
        axes[0, 1].plot(epochs, student_history['train_acc'], 'blue', linewidth=2, label='Train Accuracy')
        axes[0, 1].plot(epochs, student_history['val_acc'], 'red', linewidth=2, label='Val Accuracy')
        axes[0, 1].axhline(y=99, color='green', linestyle='--', alpha=0.7, label='Target (99%)')
        axes[0, 1].set_title('Student Accuracy Progress', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate schedule (OneCycleLR)
        if len(epochs) > 0:
            max_lr = 1e-3
            pct_start = 0.3
            total_steps = len(epochs)
            lr_schedule = []
            for epoch in epochs:
                if epoch < pct_start * total_steps:
                    # Warm-up phase
                    lr = max_lr * (epoch / (pct_start * total_steps))
                else:
                    # Decay phase
                    progress = (epoch - pct_start * total_steps) / ((1 - pct_start) * total_steps)
                    lr = max_lr * (1 - progress)
                lr_schedule.append(lr)
            
            axes[1, 0].plot(epochs, lr_schedule, 'orange', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule (OneCycleLR)', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Validation accuracy trend (with smoothing)
        if len(student_history['val_acc']) > 0:
            val_acc = student_history['val_acc']
            window = min(5, len(val_acc))
            smoothed_acc = np.convolve(val_acc, np.ones(window)/window, mode='valid')
            smoothed_epochs = list(epochs)[window-1:]
            
            axes[1, 1].plot(epochs, val_acc, 'lightblue', alpha=0.5, label='Raw Val Accuracy')
            axes[1, 1].plot(smoothed_epochs, smoothed_acc, 'darkblue', linewidth=2, label='Smoothed Trend')
            axes[1, 1].axhline(y=99, color='green', linestyle='--', alpha=0.7, label='Target (99%)')
            axes[1, 1].set_title('Validation Accuracy Trend', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'knowledge_distillation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()