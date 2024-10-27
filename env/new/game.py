import pygame
import random
import sys

# Kích thước cửa sổ trò chơi
WIDTH = 800
HEIGHT = 600
BLOCK_SIZE = 20

# Định nghĩa hướng
class Direction:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

# Định nghĩa điểm
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Lớp trò chơi
class SnakeGameAI:
    def __init__(self):
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        # Khởi tạo rắn với 3 khối
        self.snake = [Point(WIDTH // 2, HEIGHT // 2), 
                       Point(WIDTH // 2, HEIGHT // 2 + BLOCK_SIZE), 
                       Point(WIDTH // 2, HEIGHT // 2 + 2 * BLOCK_SIZE)]
        self.direction = Direction.RIGHT
        self.food = None
        self.score = 0
        self.reward = 0  # Khởi tạo reward
        self.place_food()  # Đặt trái cây

    def place_food(self):
        x = random.randint(0, (WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE
        y = random.randint(0, (HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE
        self.food = Point(x, y)

    def is_collision(self, point):
        # Kiểm tra va chạm với tường hoặc với bản thân
        if point.x < 0 or point.x >= WIDTH or point.y < 0 or point.y >= HEIGHT:
            return True
        for block in self.snake[1:]:
            if point.x == block.x and point.y == block.y:
                return True
        return False
    
    def get_state(self):
        head = self.snake[0]  # Get the head of the snake

        # Create points for the adjacent squares
        point_l = Point(head.x - BLOCK_SIZE, head.y)  # Left
        point_r = Point(head.x + BLOCK_SIZE, head.y)  # Right
        point_u = Point(head.x, head.y - BLOCK_SIZE)  # Up
        point_d = Point(head.x, head.y + BLOCK_SIZE)  # Down

        # Check for collisions with walls
        collision_l = point_l.x < 0  # Left wall
        collision_r = point_r.x >= WIDTH  # Right wall
        collision_u = point_u.y < 0  # Upper wall
        collision_d = point_d.y >= HEIGHT  # Lower wall

        # Check for collisions with itself
        collision_self = any(head.x == block.x and head.y == block.y for block in self.snake[1:])

        # Check if food is in the vicinity
        food_left = self.food.x < head.x
        food_right = self.food.x > head.x
        food_up = self.food.y < head.y
        food_down = self.food.y > head.y

        # Create the state vector
        state = [
            collision_l,  # Left wall collision
            collision_r,  # Right wall collision
            collision_u,  # Upper wall collision
            collision_d,  # Lower wall collision
            collision_self,  # Self-collision
            food_left,  # Food is to the left
            food_right,  # Food is to the right
            food_up,  # Food is above
            food_down  # Food is below
        ]
        return state

    def play_step(self, action):
        # Di chuyển rắn
        if action[0] == 1:  # LEFT
            self.direction = Direction.LEFT
        elif action[1] == 1:  # RIGHT
            self.direction = Direction.RIGHT
        elif action[2] == 1:  # UP
            self.direction = Direction.UP
        elif action[3] == 1:  # DOWN
            self.direction = Direction.DOWN

        # Tính vị trí đầu rắn mới
        head = self.snake[0]
        if self.direction == Direction.LEFT:
            new_head = Point(head.x - BLOCK_SIZE, head.y)
        elif self.direction == Direction.RIGHT:
            new_head = Point(head.x + BLOCK_SIZE, head.y)
        elif self.direction == Direction.UP:
            new_head = Point(head.x, head.y - BLOCK_SIZE)
        elif self.direction == Direction.DOWN:
            new_head = Point(head.x, head.y + BLOCK_SIZE)

        # Kiểm tra va chạm
        if self.is_collision(new_head):
            return -10, True, self.score  # Phần thưởng âm cho việc chết

        # Thêm đầu mới vào rắn
        self.snake.insert(0, new_head)

        # Kiểm tra ăn trái cây
        if new_head.x == self.food.x and new_head.y == self.food.y:
            self.score += 1
            self.reward = 10  # Phần thưởng khi ăn trái cây
            self.place_food()  # Đặt trái cây mới
        else:
            self.snake.pop()  # Xóa đuôi nếu không ăn trái cây
            self.reward = 0  # Không có phần thưởng nếu không ăn

        return self.reward, False, self.score  # Không chết

    def draw_window(self):
        self.window.fill((0, 0, 0))  # Làm sạch màn hình
        for block in self.snake:
            pygame.draw.rect(self.window, (0, 0, 255), pygame.Rect(block.x, block.y, BLOCK_SIZE, BLOCK_SIZE))  # Vẽ rắn màu xanh dương
        pygame.draw.rect(self.window, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))  # Vẽ trái cây

        # Vẽ reward lên màn hình
        font = pygame.font.SysFont('Arial', 20)  # Khởi tạo font
        reward_text = font.render(f'Reward: {self.reward}', True, (255, 255, 255))  # Màu trắng
        self.window.blit(reward_text, (10, 10))  # Vị trí trên màn hình

        pygame.display.update()  # Cập nhật màn hình

    def run(self):
        pygame.init()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Nhận input từ người dùng
            action = [0, 0, 0, 0]
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[2] = 1
            elif keys[pygame.K_DOWN]:
                action[3] = 1
            elif keys[pygame.K_LEFT]:
                action[0] = 1
            elif keys[pygame.K_RIGHT]:
                action[1] = 1

            # Chơi bước tiếp theo
            reward, game_over, score = self.play_step(action)
            self.draw_window()
            self.clock.tick(10)  # Đặt tốc độ khung hình

if __name__ == '__main__':
    game = SnakeGameAI()
    game.run()