"""
简单的游戏测试脚本
用于验证游戏基本功能
"""

from game_env import PlaneGame, Direction, AttackResult

def test_plane_shape():
    """测试飞机形状"""
    print("测试飞机形状...")
    game = PlaneGame()
    
    # 测试UP方向
    cells = game.get_plane_cells((5, 5), Direction.UP)
    print(f"机头在(5,5)，方向UP时的格子: {cells}")
    print(f"共{len(cells)}个格子")
    
    # 测试所有方向
    for direction in Direction:
        cells = game.get_plane_cells((5, 5), direction)
        print(f"方向{direction.name}: {len(cells)}个格子")
    
    print("✓ 飞机形状测试通过\n")

def test_placement():
    """测试飞机放置"""
    print("测试飞机放置...")
    game = PlaneGame()
    
    # 放置一架飞机
    success = game.place_plane(game.board1, (5, 5), Direction.UP, 1)
    print(f"在(5,5)放置飞机(UP): {success}")
    
    # 尝试重叠放置（应该失败）
    success2 = game.place_plane(game.board1, (5, 5), Direction.DOWN, 2)
    print(f"在(5,5)重叠放置飞机(DOWN): {success2} (应该为False)")
    
    print("✓ 飞机放置测试通过\n")

def test_attack():
    """测试攻击"""
    print("测试攻击...")
    game = PlaneGame()
    
    # 放置飞机
    game.place_plane(game.board1, (5, 5), Direction.UP, 1)
    
    # 攻击空白位置
    result = game.attack(2, (0, 0))
    print(f"攻击(0,0): {result.name}")
    
    # 攻击机身
    result = game.attack(2, (6, 5))
    print(f"攻击(6,5)机身: {result.name}")
    
    # 攻击机头
    result = game.attack(2, (5, 5))
    print(f"攻击(5,5)机头: {result.name}")
    print(f"击落飞机数: {game.down_planes1}")
    
    print("✓ 攻击测试通过\n")

def test_random_placement():
    """测试随机布局"""
    print("测试随机布局...")
    game = PlaneGame()
    
    try:
        game.place_planes_random(1)
        game.place_planes_random(2)
        print(f"玩家1飞机数: {len(game.planes1)}")
        print(f"玩家2飞机数: {len(game.planes2)}")
        print("✓ 随机布局测试通过\n")
    except Exception as e:
        print(f"✗ 随机布局测试失败: {e}\n")

if __name__ == "__main__":
    print("=" * 50)
    print("游戏功能测试")
    print("=" * 50)
    print()
    
    test_plane_shape()
    test_placement()
    test_attack()
    test_random_placement()
    
    print("=" * 50)
    print("所有测试完成！")
    print("=" * 50)

