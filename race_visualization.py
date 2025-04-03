import pygame
import math
import numpy as np
from f1_simulation import simulate_race

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
CAR_SIZE = 10
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

class RaceVisualization:
    def __init__(self, circuit_id):
        print("Initializing race visualization...")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("F1 Race Simulation")
        self.clock = pygame.time.Clock()
        self.circuit_id = circuit_id
        
        # Load circuit data
        print("Loading circuit data...")
        self.circuit_data = self.load_circuit_data()
        
        # Initialize race state
        print("Initializing race state...")
        self.race_state = self.initialize_race_state()
        print("Initialization complete!")
        
    def load_circuit_data(self):
        # Create a more complex track with inner and outer boundaries
        track = {
            'outer_track': self.generate_track(1.1),
            'inner_track': self.generate_track(0.9),
            'center_line': self.generate_track(1.0),
            'width': 60
        }
        return track
    
    def generate_track(self, scale):
        points = []
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2
        radius_x = 400 * scale
        radius_y = 250 * scale
        
        # Generate a more complex track shape
        for t in np.linspace(0, 2 * np.pi, 200):
            # Add some variation to make it more interesting
            r_x = radius_x * (1 + 0.1 * math.sin(3 * t))
            r_y = radius_y * (1 + 0.1 * math.cos(2 * t))
            
            x = center_x + r_x * math.cos(t)
            y = center_y + r_y * math.sin(t)
            points.append((x, y))
        
        return points
    
    def initialize_race_state(self):
        # Get race simulation data
        print("Running race simulation...")
        race_results = simulate_race(self.circuit_id)
        print(f"Simulation complete. Number of cars: {len(race_results)}")
        
        # Initialize car positions and states
        cars = []
        for i in range(len(race_results)):
            car = {
                'position': i + 1,
                'lap_time': race_results.lap_times[i],
                'points': race_results.points[i],
                'current_lap': 0,
                'progress': 0,
                'color': self.get_car_color(i)
            }
            cars.append(car)
            print(f"Car {i+1}: Lap time = {car['lap_time']:.2f}s, Points = {car['points']}")
        
        return {
            'cars': cars,
            'total_laps': 3,  # Reduced for visualization
            'current_time': 0,
            'race_finished': False
        }
    
    def get_car_color(self, index):
        # Generate different colors for different cars
        hue = (index * 137.5) % 360
        saturation = 80 + (index * 20) % 20
        value = 80 + (index * 20) % 20
        color = pygame.Color(0, 0, 0)
        color.hsva = (hue, saturation, value, 100)
        return color
    
    def update(self):
        if self.race_state['race_finished']:
            return
            
        self.race_state['current_time'] += 1/FPS
        
        # Update car positions
        all_finished = True
        for car in self.race_state['cars']:
            # Calculate progress around track
            total_progress = self.race_state['current_time'] / car['lap_time']
            car['current_lap'] = int(total_progress)
            car['progress'] = total_progress % 1
            
            if car['current_lap'] < self.race_state['total_laps']:
                all_finished = False
        
        if all_finished:
            self.race_state['race_finished'] = True
            print("Race finished!")
            # Sort cars by laps completed and progress
            self.race_state['cars'].sort(key=lambda x: (-x['current_lap'], -x['progress']))
    
    def draw(self):
        self.screen.fill(WHITE)
        
        # Draw track
        pygame.draw.lines(self.screen, GRAY, True, self.circuit_data['outer_track'], 2)
        pygame.draw.lines(self.screen, GRAY, True, self.circuit_data['inner_track'], 2)
        
        # Draw cars
        for car in self.race_state['cars']:
            # Calculate car position on track
            track_points = self.circuit_data['center_line']
            track_index = int(car['progress'] * len(track_points))
            if track_index >= len(track_points):
                track_index = len(track_points) - 1
            
            pos = track_points[track_index]
            
            # Draw car
            pygame.draw.circle(self.screen, car['color'], 
                             (int(pos[0]), int(pos[1])), CAR_SIZE)
            
            # Draw position and lap info
            font = pygame.font.Font(None, 24)
            info_text = f"P{car['position']} L{car['current_lap'] + 1}/{self.race_state['total_laps']}"
            text = font.render(info_text, True, car['color'])
            self.screen.blit(text, (pos[0] - 30, pos[1] - 20))
        
        # Draw race status
        font = pygame.font.Font(None, 36)
        time_text = f"Race Time: {self.race_state['current_time']:.1f}s"
        text = font.render(time_text, True, BLACK)
        self.screen.blit(text, (10, 10))
        
        if self.race_state['race_finished']:
            finish_text = "RACE FINISHED!"
            text = font.render(finish_text, True, RED)
            text_rect = text.get_rect(center=(WINDOW_WIDTH//2, 50))
            self.screen.blit(text, text_rect)
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    visualization = RaceVisualization(1)  # Circuit ID 1
    visualization.run() 