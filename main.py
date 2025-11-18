"""
炸飞机游戏主程序
"""

import sys
import argparse

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="炸飞机游戏")
    parser.add_argument("--mode", type=str, choices=["gui", "train"], 
                       default="gui", help="运行模式：gui（图形界面）或train（训练）")
    parser.add_argument("--episodes", type=int, default=1000, 
                       help="训练模式下的训练轮数")
    parser.add_argument("--save-interval", type=int, default=100, 
                       help="训练模式下的保存间隔")
    parser.add_argument("--model", type=str, default="ai_model.pkl", 
                       help="模型文件路径")
    parser.add_argument("--self-play", action="store_true", 
                       help="使用自我对弈模式训练")
    parser.add_argument("--gpu", action="store_true", 
                       help="使用GPU加速（注意：基于表格的方法GPU加速效果有限）")
    
    args = parser.parse_args()
    
    if args.mode == "gui":
        # 启动图形界面
        from gui import main as gui_main
        gui_main()
    elif args.mode == "train":
        # 启动训练
        from train import train_ai, train_with_self_play
        if args.self_play:
            train_with_self_play(args.episodes, args.save_interval, args.model, args.gpu)
        else:
            train_ai(args.episodes, args.save_interval, args.model, args.gpu)


if __name__ == "__main__":
    main()

