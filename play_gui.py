#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame
import torch

from aircraft_ai.alphazero_net import AlphaZeroNet
from aircraft_ai.board import Board
from aircraft_ai.constants import BOARD_SIZE, CELL_HEAD, CELL_HIT, CELL_MISS, CELL_UNKNOWN, PLANES_PER_SIDE
from aircraft_ai.env import AircraftEnv, AttackState
from aircraft_ai.mcts import MCTS


def coord_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


class PolicyWrapper:
    def __init__(self, model_path: Path, device: torch.device, mcts_simulations: int = 100) -> None:
        self.device = device
        self.network: Optional[AlphaZeroNet] = None
        self.mcts: Optional[MCTS] = None
        self.env: Optional[AircraftEnv] = None
        self.mcts_simulations = mcts_simulations
        
        if model_path.exists():
            try:
                network = AlphaZeroNet().to(device)
                checkpoint = torch.load(model_path, map_location=device)
                
                # 支持新格式（包含model_state_dict）和旧格式（直接是state_dict）
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    network.load_state_dict(checkpoint["model_state_dict"])
                    print(f"Loaded checkpoint from iteration {checkpoint.get('iteration', 'unknown')}, "
                          f"win_rate: {checkpoint.get('win_rate', 0.0):.2%}")
                else:
                    # 旧格式：直接是state_dict
                    network.load_state_dict(checkpoint)
                
                network.eval()
                self.network = network
                
                # 创建临时环境用于MCTS
                self.env = AircraftEnv()
                
                # 创建MCTS
                def step_fn(state: np.ndarray, action: int) -> tuple:
                    feedback = self.env.step(action)
                    return feedback.observation, feedback.reward, feedback.done
                
                self.mcts = MCTS(
                    network=network,
                    step_fn=step_fn,
                    c_puct=1.0,
                    num_simulations=mcts_simulations,
                    temperature=0.0,  # 评估时使用确定性选择
                    device=device,
                )
                print(f"Model loaded: {model_path}")
            except Exception as exc:
                print(f"Failed to load model, using random policy: {exc}")
        else:
            print(f"Model not found at {model_path}, using random policy.")

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise ValueError("No valid actions available")
        
        if self.network is None or self.mcts is None:
            return int(np.random.choice(valid))
        
        # 使用MCTS搜索（快速模式，较少模拟次数）
        self.mcts.num_simulations = self.mcts_simulations
        action, _ = self.mcts.search(obs, mask)
        return action


class BattleSession:
    def __init__(self, policy: PolicyWrapper, seed: int = 1) -> None:
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.auto_mode = False
        self.last_move: Optional[Tuple[str, int, int, object]] = None
        self.reset()

    def reset(self) -> None:
        self.ai_board = Board(seed=int(self.rng.integers(0, 2**31 - 1)))
        self.player_board = Board(seed=int(self.rng.integers(0, 2**31 - 1)))
        self.player_state = AttackState(self.ai_board)
        self.player_state.reset_memory()
        self.ai_state = AttackState(self.player_board)
        self.ai_state.reset_memory()
        self.current_turn = "player"
        self.winner: Optional[str] = None
        self.last_move = None

    def toggle_auto(self) -> None:
        self.auto_mode = not self.auto_mode

    def human_attack(self, row: int, col: int) -> None:
        if self.auto_mode or self.winner or self.current_turn != "player":
            return
        if self.player_state.knowledge[row, col] != CELL_UNKNOWN:
            return
        action = row * BOARD_SIZE + col
        feedback = self.player_state.apply_action(action)
        self.last_move = ("Player", row, col, feedback.info.get("result"))
        if feedback.done:
            self.winner = "Player"
            return
        self.current_turn = "ai"
        self._ai_turn()

    def _ai_turn(self) -> None:
        if self.winner:
            return
        obs = self.ai_state.observation()
        mask = self.ai_state.action_mask()
        if mask.sum() == 0:
            self.winner = "Player" if not self.auto_mode else "Left AI"
            return
        action = self.policy.act(obs, mask)
        feedback = self.ai_state.apply_action(action)
        row, col = divmod(action, BOARD_SIZE)
        self.last_move = ("AI", row, col, feedback.info.get("result"))
        if feedback.done:
            self.winner = "AI" if not self.auto_mode else "Right AI"
        else:
            self.current_turn = "player"

    def auto_tick(self) -> None:
        if not self.auto_mode or self.winner:
            return
        if self.current_turn == "player":
            obs = self.player_state.observation()
            mask = self.player_state.action_mask()
            if mask.sum() == 0:
                self.winner = "Right AI"
                return
            action = self.policy.act(obs, mask)
            feedback = self.player_state.apply_action(action)
            row, col = divmod(action, BOARD_SIZE)
            self.last_move = ("Left AI", row, col, feedback.info.get("result"))
            if feedback.done:
                self.winner = "Left AI"
            else:
                self.current_turn = "ai"
        else:
            self._ai_turn()

    def remaining_ai_planes(self) -> int:
        return self.ai_board.remaining_planes()

    def remaining_player_planes(self) -> int:
        return self.player_board.remaining_planes()


class GameUI:
    GRID_COLOR = (84, 110, 122)
    UNKNOWN_COLOR = (117, 117, 117)
    MISS_COLOR = (69, 90, 100)
    HIT_COLOR = (102, 187, 106)
    HEAD_COLOR = (229, 57, 53)
    BG_COLOR = (21, 25, 34)
    PANEL_BG = (33, 43, 54)
    TEXT_COLOR = (236, 239, 241)
    PLANE_HINT = (64, 128, 255)

    def __init__(self, session: BattleSession, fps: int = 60) -> None:
        pygame.init()
        self.session = session
        self.cell_size = 40
        self.margin = 40
        self.gap = 80
        width = self.cell_size * BOARD_SIZE * 2 + self.gap + self.margin * 2
        height = self.cell_size * BOARD_SIZE + self.margin * 2 + 160
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Aircraft Battle - Deep RL Opponent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.fps = fps
        self.auto_timer = 0.0
        self.auto_interval = 0.4

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(self.fps) / 1000.0
            self.auto_timer += dt
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.session.reset()
                    elif event.key == pygame.K_SPACE:
                        self.session.toggle_auto()
                        self.auto_timer = 0.0

            if self.session.auto_mode and self.auto_timer >= self.auto_interval:
                self.session.auto_tick()
                self.auto_timer = 0.0

            self.draw()

        pygame.quit()

    def handle_click(self, pos: Tuple[int, int]) -> None:
        if self.session.auto_mode or self.session.winner:
            return
        grid_rect = self._grid_rect(left=True)
        if not grid_rect.collidepoint(pos):
            return
        rel_x = pos[0] - grid_rect.x
        rel_y = pos[1] - grid_rect.y
        col = rel_x // self.cell_size
        row = rel_y // self.cell_size
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            self.session.human_attack(int(row), int(col))

    def _grid_rect(self, left: bool) -> pygame.Rect:
        x = self.margin if left else self.margin + self.cell_size * BOARD_SIZE + self.gap
        y = self.margin
        return pygame.Rect(x, y, self.cell_size * BOARD_SIZE, self.cell_size * BOARD_SIZE)

    def draw(self) -> None:
        self.screen.fill(self.BG_COLOR)
        self._draw_panel(self._grid_rect(True), self.session.player_state.knowledge, False)
        self._draw_panel(
            self._grid_rect(False),
            self.session.ai_state.knowledge,
            True,
            self.session.player_board,
        )
        self._draw_labels()
        pygame.display.flip()

    def _draw_panel(self, rect: pygame.Rect, knowledge: np.ndarray, show_planes: bool, board: Optional[Board] = None) -> None:
        pygame.draw.rect(self.screen, self.PANEL_BG, rect)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell_rect = pygame.Rect(
                    rect.x + col * self.cell_size,
                    rect.y + row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                value = knowledge[row, col]
                color = self.UNKNOWN_COLOR
                if value == CELL_MISS:
                    color = self.MISS_COLOR
                elif value == CELL_HIT:
                    color = self.HIT_COLOR
                elif value == CELL_HEAD:
                    color = self.HEAD_COLOR
                pygame.draw.rect(self.screen, color, cell_rect.inflate(-2, -2))
                pygame.draw.rect(self.screen, self.GRID_COLOR, cell_rect, 1)

        if show_planes and board is not None:
            for (x, y) in board.disclosed_cells().keys():
                cell_rect = pygame.Rect(
                    rect.x + x * self.cell_size,
                    rect.y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, self.PLANE_HINT, cell_rect.inflate(-8, -8), 2)

    def _draw_labels(self) -> None:
        left_text = f"Player Attack Panel | Destroyed: {PLANES_PER_SIDE - self.session.remaining_ai_planes()} / {PLANES_PER_SIDE}"
        right_text = f"AI Attack Panel | Player Remaining: {self.session.remaining_player_planes()} planes"
        lt = self.font.render(left_text, True, self.TEXT_COLOR)
        rt = self.font.render(right_text, True, self.TEXT_COLOR)
        self.screen.blit(lt, (self._grid_rect(True).x, self._grid_rect(True).y - 30))
        self.screen.blit(rt, (self._grid_rect(False).x, self._grid_rect(False).y - 30))

        instructions = "[Mouse] Attack  [R] Reset  [Space] AI Self-Play"
        inst_surface = self.font.render(instructions, True, self.TEXT_COLOR)
        self.screen.blit(inst_surface, (self.margin, self._grid_rect(True).bottom + 20))

        mode_text = "Self-Play: ON" if self.session.auto_mode else "Self-Play: OFF"
        status_text = mode_text
        if self.session.winner:
            status_text = f"Winner: {self.session.winner}"
        status_surface = self.font.render(status_text, True, self.TEXT_COLOR)
        self.screen.blit(status_surface, (self.margin, self._grid_rect(True).bottom + 60))

        if self.session.last_move:
            actor, row, col, result = self.session.last_move
            desc = "Miss"
            if result:
                if result.outcome.name == "HEAD":
                    desc = "Head Shot"
                elif result.outcome.name == "HIT":
                    desc = "Body Hit"
                else:
                    desc = "Miss"
            move_text = f"Last Move: {actor} -> {coord_label(row, col)} ({desc})"
            move_surface = self.small_font.render(move_text, True, self.TEXT_COLOR)
            self.screen.blit(move_surface, (self.margin, self._grid_rect(True).bottom + 95))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aircraft Battle GUI")
    parser.add_argument("--model", type=Path, default=None, help="Model path (default: try best, then latest, then main)")
    parser.add_argument("--fps", type=int, default=60, help="Frame rate")
    parser.add_argument("--mcts-simulations", type=int, default=100, help="MCTS simulations per move (lower=faster)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 如果没有指定模型路径，尝试按优先级加载：最佳模型 -> 最新模型 -> 主模型
    if args.model is None:
        checkpoint_dir = Path("artifacts/checkpoints")
        best_model = checkpoint_dir / "checkpoint_best.pt"
        latest_model = checkpoint_dir / "checkpoint_latest.pt"
        main_model = Path("artifacts/alphazero.pt")
        
        if best_model.exists():
            args.model = best_model
            print(f"Using best checkpoint: {best_model}")
        elif latest_model.exists():
            args.model = latest_model
            print(f"Using latest checkpoint: {latest_model}")
        else:
            args.model = main_model
            print(f"Using main model: {main_model}")
    
    policy = PolicyWrapper(args.model, device, mcts_simulations=args.mcts_simulations)
    session = BattleSession(policy)
    GameUI(session, fps=args.fps).run()


if __name__ == "__main__":
    main()
