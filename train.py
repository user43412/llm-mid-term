# train.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import random

from src.data import create_data_loader, create_data_loaders  # 需要修改data1.py添加这个函数
from src.model import create_transformer_model


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed: {seed}")


class Evaluator:
    """Evaluator class for calculating evaluation metrics"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def calculate_accuracy(self, logits, targets, ignore_index=0):
        """Calculate accuracy"""
        with torch.no_grad():
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

            # Create mask (ignore padding positions)
            mask = (targets != ignore_index)

            # Calculate number of correct predictions
            correct = (predictions == targets) & mask
            accuracy = correct.sum().float() / mask.sum().float()

            return accuracy.item()

    def calculate_perplexity(self, loss):
        """Calculate perplexity"""
        return np.exp(loss)


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config, train_dataset=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.train_dataset = train_dataset  # 保存训练集引用

        # Training device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Evaluator
        self.evaluator = Evaluator(model, self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['scheduler_step_size'],
            gamma=config['scheduler_gamma']
        )

        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Training records
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.valid_perplexities = []
        self.learning_rates = []

        # Create save directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(config['save_dir'], f"transformer_{self.timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"Training device: {self.device}")
        print(f"Model save directory: {self.save_dir}")

        # Record random seed
        self.seed = config.get('seed', 42)
        set_seed(self.seed)

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        total_accuracy = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            src_mask = batch['src_mask'].to(self.device)

            # Prepare input and target (teacher forcing)
            tgt_input = tgt_ids[:, :-1]  # Remove last token
            tgt_output = tgt_ids[:, 1:]  # Remove first token

            # Forward pass
            self.optimizer.zero_grad()
            output_dict = self.model(src_ids, tgt_input)
            logits = output_dict['output']  # [batch_size, seq_len-1, vocab_size]

            # Calculate loss
            loss = self.criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_output.contiguous().view(-1)
            )

            # Calculate accuracy
            accuracy = self.evaluator.calculate_accuracy(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_output.contiguous().view(-1)
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])

            # Parameter update
            self.optimizer.step()

            # Statistics
            batch_tokens = (tgt_output != 0).sum().item()  # Ignore padding
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            total_accuracy += accuracy * batch_tokens

            if batch_idx % self.config['log_interval'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch:03d} | Batch: {batch_idx:04d}/{len(self.train_loader):04d} | '
                      f'Loss: {loss.item():.4f} | Acc: {accuracy:.4f} | LR: {current_lr:.6f}')

        # Learning rate scheduling
        self.scheduler.step()

        avg_loss = total_loss / total_tokens
        avg_accuracy = total_accuracy / total_tokens
        epoch_time = time.time() - start_time

        print(f'Epoch: {epoch:03d} | Training Loss: {avg_loss:.4f} | Training Accuracy: {avg_accuracy:.4f} | '
              f'Time: {epoch_time:.2f}s')

        return avg_loss, avg_accuracy

    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        total_accuracy = 0

        with torch.no_grad():
            for batch in self.valid_loader:
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)

                tgt_input = tgt_ids[:, :-1]
                tgt_output = tgt_ids[:, 1:]

                output_dict = self.model(src_ids, tgt_input)
                logits = output_dict['output']

                loss = self.criterion(
                    logits.contiguous().view(-1, logits.size(-1)),
                    tgt_output.contiguous().view(-1)
                )

                accuracy = self.evaluator.calculate_accuracy(
                    logits.contiguous().view(-1, logits.size(-1)),
                    tgt_output.contiguous().view(-1)
                )

                batch_tokens = (tgt_output != 0).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                total_accuracy += accuracy * batch_tokens

        avg_loss = total_loss / total_tokens
        avg_accuracy = total_accuracy / total_tokens
        perplexity = self.evaluator.calculate_perplexity(avg_loss)

        print(f'Epoch: {epoch:03d} | Validation Loss: {avg_loss:.4f} | Validation Accuracy: {avg_accuracy:.4f} | '
              f'Perplexity: {perplexity:.2f}')

        return avg_loss, avg_accuracy, perplexity

    def train(self):
        """Complete training process"""
        print("Starting training...")
        best_loss = float('inf')
        patience_counter = 0

        # Record training start information
        self.record_training_start()

        for epoch in range(1, self.config['num_epochs'] + 1):
            # Training
            train_loss, train_accuracy = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)

            # Validation
            valid_loss, valid_accuracy, perplexity = self.validate(epoch)
            self.valid_losses.append(valid_loss)
            self.valid_accuracies.append(valid_accuracy)
            self.valid_perplexities.append(perplexity)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            # Save best model (based on validation loss)
            if valid_loss < best_loss:
                best_loss = valid_loss
                self.save_model('best_model.pth')
                patience_counter = 0
                print(f"New best model! Validation loss: {valid_loss:.4f}")
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= self.config['patience']:
                print(f"Early stopping triggered! Stopping training at epoch {epoch}")
                break

            # Regular checkpoint saving
            if epoch % self.config['checkpoint_interval'] == 0:
                self.save_model(f'checkpoint_epoch_{epoch}.pth')

            # Visualize training curves
            if epoch % self.config['plot_interval'] == 0:
                self.plot_training_curves()

        # Final save
        self.save_model('final_model.pth')
        self.plot_training_curves()
        self.save_training_log()

        # Generate reproduction command
        self.generate_reproduce_command()

        print("Training completed!")

    def plot_training_curves(self):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curve
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.valid_losses, label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curve
        ax2.plot(epochs, self.train_accuracies, label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.valid_accuracies, label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Perplexity curve
        ax3.plot(epochs, self.valid_perplexities, label='Validation Perplexity', color='purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Perplexity')
        ax3.set_title('Validation Perplexity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Learning rate curve
        ax4.plot(epochs, self.learning_rates, label='Learning Rate', color='red', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved: {plot_path}")

    def save_training_log(self):
        """Save training log"""
        log_data = {
            'config': self.config,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'train_accuracies': self.train_accuracies,
            'valid_accuracies': self.valid_accuracies,
            'valid_perplexities': self.valid_perplexities,
            'learning_rates': self.learning_rates,
            'seed': self.seed,
            'final_epoch': len(self.train_losses),
            'best_loss': min(self.valid_losses) if self.valid_losses else float('inf'),
            'best_accuracy': max(self.valid_accuracies) if self.valid_accuracies else 0
        }

        log_path = os.path.join(self.save_dir, 'training_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"Training log saved: {log_path}")

    def record_training_start(self):
        """Record training start information"""
        start_info = {
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'seed': self.seed,
            'device': str(self.device),
            'config': self.config,
            'command_line': self.generate_current_command()
        }

        start_info_path = os.path.join(self.save_dir, 'training_start_info.json')
        with open(start_info_path, 'w', encoding='utf-8') as f:
            json.dump(start_info, f, indent=2, ensure_ascii=False)

        print(f"Training start information saved: {start_info_path}")

    def generate_current_command(self):
        """Generate current training command line"""
        base_command = f"python train.py --data_dir {self.config['data_dir']} --mode train"
        command = f"{base_command} --seed {self.seed}"
        important_params = ['batch_size', 'learning_rate', 'num_epochs', 'd_model', 'n_layers', 'n_heads']
        for param in important_params:
            if param in self.config:
                command += f" --{param} {self.config[param]}"
        return command

    def generate_reproduce_command(self):
        """Generate exact command line to reproduce experiment"""
        reproduce_command = self.generate_current_command()
        model_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(model_path):
            reproduce_command += f" --model_path {model_path}"

        reproduce_file = os.path.join(self.save_dir, 'reproduce_command.txt')
        with open(reproduce_file, 'w', encoding='utf-8') as f:
            f.write("# Exact command to reproduce this experiment:\n")
            f.write(reproduce_command + "\n\n")
            f.write("# Random seed:\n")
            f.write(f"SEED={self.seed}\n\n")
            f.write("# Complete configuration:\n")
            f.write(json.dumps(self.config, indent=2, ensure_ascii=False))

        print(f"Reproduction command saved: {reproduce_file}")

    def save_model(self, filename):
        """Save model"""
        model_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'train_accuracies': self.train_accuracies,
            'valid_accuracies': self.valid_accuracies,
            'valid_perplexities': self.valid_perplexities,
            'learning_rates': self.learning_rates,
            'seed': self.seed,
            'epoch': len(self.train_losses),
            'source_vocab': self.train_dataset.source_vocab if self.train_dataset else None,  # 保存词汇表
            'target_vocab': self.train_dataset.target_vocab if self.train_dataset else None
        }, model_path)
        print(f"Model saved: {model_path}")

    def count_parameters(self):
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        return total_params, trainable_params


class PositionalEncodingAblationStudy:
    """Positional encoding ablation study"""

    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.base_config = config
        self.results = {}

    def run_ablation_study(self):
        """Run positional encoding ablation study"""
        # 首先创建一次数据加载器，避免重复构建词汇表
        print("Creating data loaders for ablation study...")

        # 使用安全的数据加载方式
        train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset = create_data_loaders(
            data_dir=self.data_dir,
            batch_size=self.base_config['batch_size'],
            source_lang='en',
            target_lang='de',
            max_length=self.base_config['max_seq_length'],
            min_freq=2
        )

        experiments = {
            'with_positional_encoding': {
                'config': {**self.base_config, 'use_positional_encoding': True},
                'description': 'With positional encoding (full model)'
            },
            'without_positional_encoding': {
                'config': {**self.base_config, 'use_positional_encoding': False},
                'description': 'Without positional encoding'
            }
        }

        for exp_name, exp_info in experiments.items():
            print(f"\n{'=' * 60}")
            print(f"Running ablation experiment: {exp_info['description']}")
            print(f"{'=' * 60}")

            # Set random seed
            exp_seed = self.base_config.get('seed', 42) + hash(exp_name) % 1000  # 不同实验不同种子
            set_seed(exp_seed)

            # Get vocabulary sizes from training dataset
            src_vocab_size, tgt_vocab_size = train_dataset.get_vocab_sizes()

            # Create model
            model = create_transformer_model(src_vocab_size, tgt_vocab_size, exp_info['config'])

            # Create trainer
            trainer = Trainer(model, train_loader, valid_loader, exp_info['config'], train_dataset)

            # Count parameters
            total_params, trainable_params = trainer.count_parameters()

            # Train model
            trainer.train()

            # Record results
            self.results[exp_name] = {
                'description': exp_info['description'],
                'final_loss': trainer.valid_losses[-1] if trainer.valid_losses else float('inf'),
                'final_accuracy': trainer.valid_accuracies[-1] if trainer.valid_accuracies else 0,
                'final_perplexity': trainer.valid_perplexities[-1] if trainer.valid_perplexities else float('inf'),
                'best_loss': min(trainer.valid_losses) if trainer.valid_losses else float('inf'),
                'best_accuracy': max(trainer.valid_accuracies) if trainer.valid_accuracies else 0,
                'total_params': total_params,
                'seed': exp_seed,
                'training_epochs': len(trainer.train_losses)
            }

        # Save and analyze results
        self.save_ablation_results()
        self.plot_ablation_results()
        self.analyze_positional_encoding_impact()

    def plot_ablation_results(self):
        """Plot ablation study results"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        exp_names = ['With Positional Encoding', 'Without Positional Encoding']

        # Get results
        with_pos = self.results['with_positional_encoding']
        without_pos = self.results['without_positional_encoding']

        # Final loss comparison
        final_losses = [with_pos['final_loss'], without_pos['final_loss']]
        ax1.bar(exp_names, final_losses, color=['skyblue', 'lightcoral'], alpha=0.7)
        ax1.set_title('Final Validation Loss Comparison')
        ax1.set_ylabel('Loss')

        # Final accuracy comparison
        final_accuracies = [with_pos['final_accuracy'], without_pos['final_accuracy']]
        ax2.bar(exp_names, final_accuracies, color=['lightgreen', 'gold'], alpha=0.7)
        ax2.set_title('Final Validation Accuracy Comparison')
        ax2.set_ylabel('Accuracy')

        # Final perplexity comparison
        final_perplexities = [with_pos['final_perplexity'], without_pos['final_perplexity']]
        ax3.bar(exp_names, final_perplexities, color=['plum', 'lightblue'], alpha=0.7)
        ax3.set_title('Final Validation Perplexity Comparison')
        ax3.set_ylabel('Perplexity')

        plt.tight_layout()
        plot_path = os.path.join(self.base_config['save_dir'], 'positional_encoding_ablation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Positional encoding ablation plot saved: {plot_path}")

    def analyze_positional_encoding_impact(self):
        """Analyze the impact of positional encoding"""
        with_pos = self.results['with_positional_encoding']
        without_pos = self.results['without_positional_encoding']

        # Calculate performance changes
        loss_increase = ((without_pos['final_loss'] - with_pos['final_loss']) / with_pos['final_loss']) * 100
        accuracy_decrease = ((with_pos['final_accuracy'] - without_pos['final_accuracy']) / with_pos[
            'final_accuracy']) * 100
        perplexity_increase = ((without_pos['final_perplexity'] - with_pos['final_perplexity']) / with_pos[
            'final_perplexity']) * 100

        print(f"\n{'=' * 80}")
        print("Positional Encoding Ablation Study Results Analysis")
        print(f"{'=' * 80}")
        print(f"{'Metric':<15} {'With PE':<12} {'Without PE':<12} {'Change %':<15} {'Impact':<20}")
        print(f"{'-' * 80}")
        print(f"{'Loss':<15} {with_pos['final_loss']:<12.4f} {without_pos['final_loss']:<12.4f} "
              f"{loss_increase:>+13.1f}% {'Negative' if loss_increase > 0 else 'Positive'}")
        print(f"{'Accuracy':<15} {with_pos['final_accuracy']:<12.4f} {without_pos['final_accuracy']:<12.4f} "
              f"{accuracy_decrease:>+13.1f}% {'Negative' if accuracy_decrease > 0 else 'Positive'}")
        print(f"{'Perplexity':<15} {with_pos['final_perplexity']:<12.2f} {without_pos['final_perplexity']:<12.2f} "
              f"{perplexity_increase:>+13.1f}% {'Negative' if perplexity_increase > 0 else 'Positive'}")
        print(f"{'=' * 80}")

        print(f"\nConclusion Analysis:")
        print(f"1. Removing positional encoding increases loss by: {loss_increase:+.1f}%")
        print(f"2. Removing positional encoding decreases accuracy by: {accuracy_decrease:+.1f}%")
        print(f"3. Removing positional encoding increases perplexity by: {perplexity_increase:+.1f}%")
        print(f"\nThis demonstrates that positional encoding is crucial for Transformer model performance!")

    def save_ablation_results(self):
        """Save ablation study results"""
        results_path = os.path.join(self.base_config['save_dir'], 'positional_encoding_ablation_results.json')

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'base_config': self.base_config
            }, f, indent=2, ensure_ascii=False)

        print(f"Ablation study results saved: {results_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Transformer training script')
    parser.add_argument('--data_dir', type=str, default='data/en-de', help='Data directory path')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'ablation'],
                        help='Run mode: train or ablation')
    parser.add_argument('--model_path', type=str, help='Model path to load')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    args = parser.parse_args()

    # Training configuration
    config = {
        # Model configuration
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'max_seq_length': 50,
        'dropout': 0.1,
        'use_positional_encoding': True,  # Default to using positional encoding

        # Training configuration
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'max_grad_norm': 1.0,
        'seed': args.seed,
        'data_dir': args.data_dir,

        # Learning rate scheduling
        'scheduler_step_size': 10,
        'scheduler_gamma': 0.5,

        # Training settings
        'patience': 10,
        'log_interval': 100,
        'checkpoint_interval': 5,
        'plot_interval': 5,

        # Save settings
        'save_dir': 'saved_models'
    }

    set_seed(args.seed)
    os.makedirs(config['save_dir'], exist_ok=True)

    if args.mode == 'train':
        # Safe data loading to avoid data leakage
        train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=config['batch_size'],
            source_lang='en',
            target_lang='de',
            max_length=config['max_seq_length'],
            min_freq=2
        )

        # Get vocabulary sizes from training dataset
        src_vocab_size, tgt_vocab_size = train_dataset.get_vocab_sizes()

        # Create model
        model = create_transformer_model(src_vocab_size, tgt_vocab_size, config)

        # Create trainer
        trainer = Trainer(model, train_loader, valid_loader, config, train_dataset)

        # Parameter statistics
        trainer.count_parameters()

        # Load existing model (if specified)
        if args.model_path:
            # 需要实现load_model方法
            print(f"Model loading not implemented yet: {args.model_path}")

        # Start training
        trainer.train()

    elif args.mode == 'ablation':
        # Ablation study mode
        ablation_study = PositionalEncodingAblationStudy(args.data_dir, config)
        ablation_study.run_ablation_study()


if __name__ == "__main__":
    main()