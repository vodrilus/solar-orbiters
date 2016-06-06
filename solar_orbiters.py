"""Solar orbiters

A numerical simulation of the Solar system up to Jupiter.
Also, some random Trojans... or were they Greeks?
"""

import sys
import os
os.environ["PYSDL2_DLL_PATH"] = "c:\\Python27\\DLLs"
import sdl2
import sdl2.ext
from math import *
from random import *
import xml.etree.ElementTree as ET

ASTRO_OBJECTS_XML_PATH = 'astro_objects.xml'
WINDOW_SIZE = 800, 600
WINDOW_SCALE = 150000000000 // 50 # 150 exp 9 m = about 1 AU = 50 pixels

BLACK  = sdl2.ext.Color(0, 0, 0)
WHITE  = sdl2.ext.Color(255, 255, 255)
BLUE   = sdl2.ext.Color(0, 0, 255)
YELLOW = sdl2.ext.Color(255, 255, 0)
GRAY   = sdl2.ext.Color(150, 150, 150)

GRAV_CONSTANT = 6.67408 * 10 ** (-11) # For meters!

TIME_STEP = 3600 # The computation step in seconds
TIME_SCALE = 20 # Computational steps per frame (supposedly 10 ms per frame).
THETA = 1.0 # Distance threshold ratio. Large values increase speed but
            # sacrifice accuracy.




class SpriteMovementSystem(sdl2.ext.Applicator):
    def __init__(self):
        super(SpriteMovementSystem, self).__init__()
        self.componenttypes = Position, sdl2.ext.Sprite

    def process(self, world, componentsets):
        """Move sprites to represent planet movement"""
        for position, sprite in componentsets:
            swidth, sheight = sprite.size
            sprite.x, sprite.y = world_coord_to_screen_coord(position.x,
                                                   position.y,
                                                   camera)
            sprite.x -= swidth
            sprite.y -= sheight
        

class MovementSystem(sdl2.ext.Applicator):
    def __init__(self):
        super(MovementSystem, self).__init__()
        self.componenttypes = (Mass, Position, Velocity, Acceleration)

    def process(self, world, componentsets):
        """Apply Barnes-Hut gravity algorithm (O(n log n))"""
        # Squeeze some efficiency with local variables
        time_step = TIME_STEP
        grav_constant = GRAV_CONSTANT
        # Super clunky, but componentsets is apparently an iterator, not a list!
        comps = []
        for comptuple in componentsets:
            comps.append(comptuple)
        for i in xrange(TIME_SCALE):
            grav_data_tuples = [(mass.mass, position.x, position.y) for
                        mass, position, velocity, acceleration in
                        comps]

            root_node = QuadNode(grav_data_tuples, 10**12, 0, 0)
            
            for mass, position, velocity, acceleration in comps:
                
                # Compute gravitational acceleration for step
                # Squeeze a little more efficiency with local vars
                
                ax, ay = 0, 0
                x1, y1 = position.x, position.y
                ax, ay = root_node.get_gravity_at_point(x1, y1)
                
                velocity.vx += ax * time_step
                velocity.vy += ay * time_step
                
                position.x += velocity.vx * time_step
                position.y += velocity.vy * time_step

                

class QuadNode():
    def __init__(self, data_tuples, width, x, y):
        # data_tuples = [(mass, x, y)...]
        self.width = width
        self.mass = 0
        self.center_of_gravity_x = 0
        self.center_of_gravity_y = 0
        self.is_internal = True
        
        length = len(data_tuples)
        if length > 1:
            nw_list = []
            ne_list = []
            sw_list = []
            se_list = []
            x_by_mass = 0
            y_by_mass = 0
            
            for o in data_tuples:
                object_mass = o[0]
                self.mass += object_mass
                object_x = o[1]
                object_y = o[2]
                x_by_mass += object_x * object_mass
                y_by_mass += object_y * object_mass
                if object_y <= y: # Yes, this feels messed up - screen coords...
                    if object_x <= x:
                        nw_list.append(o)
                    else:
                        ne_list.append(o)
                else:
                    if object_x <= x:
                        sw_list.append(o)
                    else:
                        se_list.append(o)

            self.center_of_gravity_x = x_by_mass / self.mass
            self.center_of_gravity_y = y_by_mass / self.mass
            
            halfwidth = width/2
            self.nw = QuadNode(nw_list, halfwidth, x - halfwidth, y - halfwidth)
            self.ne = QuadNode(ne_list, halfwidth, x + halfwidth, y - halfwidth)
            self.sw = QuadNode(sw_list, halfwidth, x - halfwidth, y + halfwidth)
            self.se = QuadNode(se_list, halfwidth, x + halfwidth, y + halfwidth)
            # Is it really necessary to define the direction of the node? No?

                
        elif length == 1:
            self.is_internal = False
            astro_object = data_tuples[0]
            self.mass = astro_object[0]
            self.center_of_gravity_x = astro_object[1]
            self.center_of_gravity_y = astro_object[2]
            
        else:
            self.is_internal = False
            self.center_of_gravity = 0, 0
            

    def get_gravity_at_point(self, x, y):
        if self.mass == 0:
            return 0.0, 0.0
        elif self.is_accurate_enough(x, y):
            if self.center_of_gravity_x == x and \
               self.center_of_gravity_y == y:
                return 0.0, 0.0
            cog_x = self.center_of_gravity_x
            cog_y = self.center_of_gravity_y

            # Gravitational acceleration generated at this location
            # by given object, as per Newton's gravitational equation,
            # but divided on both sides by the mass of the affected
            # object. (Eigenvalue)
            a = GRAV_CONSTANT * self.mass / ((cog_x - x)**2 +
                                             (cog_y - y)**2)
            
            # Direction of the acceleration vector
            direction = atan2(cog_y - y, cog_x - x)
            # X and y components of the gravity vector
            ax = a * cos(direction)
            ay = a * sin(direction)
            return ax, ay
        else:
            child_nodes = [self.nw, self.ne, self.sw, self.se]
            child_gravities = [n.get_gravity_at_point(x,y) for n in child_nodes]
            ax, ay = 0, 0
            for n in child_nodes:
                child_ax, child_ay = n.get_gravity_at_point(x,y)
                ax += child_ax
                ay += child_ay
            return ax, ay


    def is_accurate_enough(self, x, y):
        if self.is_internal:
            distance = sqrt((self.center_of_gravity_x - x) ** 2 +
                            (self.center_of_gravity_y - y) ** 2)
            if self.width / distance <= THETA:
                return True
            else:
                return False
        else:
            return True

# The main renderer
class SoftwareRenderSystem(sdl2.ext.SoftwareSpriteRenderSystem):
    def __init__(self, window):
        super(SoftwareRenderSystem, self).__init__(window)

    def render(self, components):
        sdl2.ext.fill(self.surface, BLACK)
        super(SoftwareRenderSystem, self).render(components)


# Just in case the Software Renderer is unavailable for some reason.
class TextureRenderSystem(sdl2.ext.TextureSpriteRenderSystem):
    def __init__(self, renderer):
        super(TextureRenderSystem, self).__init__(renderer)
        self.renderer = renderer

    def render(self, components):
        tmp = self.renderer.color
        self.renderer.color = BLACK
        self.renderer.clear()
        self.renderer.color = tmp
        super(TextureRenderSystem, self).render(components)


class Camera():
    """Allows movement of display."""
    def __init__(self):
        self.x = 0
        self.y = 0
        self.scale = WINDOW_SCALE + 0 # Copy, not reference!

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def zoom(self, factor):
        self.scale = int(self.scale / factor)


def world_coord_to_screen_coord(x_world,y_world,camera):
    w_width, w_height = WINDOW_SIZE
    x_screen = int(w_width / 2 + x_world / camera.scale) - camera.x
    y_screen = int(w_height / 2+ y_world / camera.scale) - camera.y
    return x_screen, y_screen


# Define data bags
class Mass(object):
    def __init__(self):
        super(Mass, self).__init__()
        self.mass = 0

class Position(object):
    def __init__(self):
        super(Position, self).__init__()
        self.x = 0
        self.y = 0

class Velocity(object):
    def __init__(self):
        super(Velocity, self).__init__()
        self.vx = 0
        self.vy = 0

class Acceleration(object):
    def __init__(self):
        super(Acceleration, self).__init__()
        self.ax = 0
        self.ay = 0
        

class AstronomicalObject(sdl2.ext.Entity):
    def __init__(self, world, sprite, mass=0, posx=0, posy=0, vx=0, vy=0):
        self.sprite = sprite
        self.sprite.position = world_coord_to_screen_coord(posx,posy,camera)

        self.position = Position()
        self.position.x = posx
        self.position.y = posy

        self.velocity = Velocity()
        self.velocity.vx = vx
        self.velocity.vy = vy
        
        self.acceleration = Acceleration()
        self.mass = Mass()
        self.mass.mass = mass

def apply_gravity(gravitational_objects):
    # Squeeze some efficiency with local variables
    time_step = TIME_STEP
    grav_constant = GRAV_CONSTANT

    for i in xrange(TIME_SCALE):
        for o1 in gravitational_objects:
            position = o1.position
            velocity = o1.velocity
            acceleration = o1.acceleration
            sprite = o1.sprite
            
            # Compute gravitational acceleration for step
            # Squeeze a little more efficiency with local vars
            ax = acceleration.ax
            ay = acceleration.ay
            ax0 = ax + 0
            ay0 = ay + 0
            ax, ay = 0, 0
            x1, y1 = position.x, position.y
            for o2 in gravitational_objects:
                x2, y2 = o2.position.x, o2.position.y
                if x1 == x2 and y1 == y2 or o2.mass.mass < 100000:
                    # Get rid of all those nasty div by zero errors
                    # and disregard objects below 100 000 kg
                    continue
                # Gravitational acceleration generated at this location
                # by given object, as per Newton's gravitational equation,
                # but divided on both sides by the mass of the affected
                # object. (Eigenvalue)
                a = grav_constant * o2.mass.mass / ((x2 - x1)**2 +
                                                  (y2 - y1)**2)
                # Direction of the acceleration vector
                direction = atan2(y2 - y1, x2 - x1)
                # X and y components of the gravity vector
                ax += a * cos(direction)
                ay += a * sin(direction)

            if ax0 == 0 and ay0 == 0:
                velocity.vx += ax * time_step
                velocity.vy += ay * time_step
                
                position.x += velocity.vx * time_step
                position.y += velocity.vy * time_step
            else:
                velocity.vx += ax0 * time_step / 2
                velocity.vy += ay0 * time_step / 2
                
                position.x += velocity.vx * time_step
                position.y += velocity.vy * time_step

                velocity.vx += ax * time_step / 2
                velocity.vy += ay * time_step / 2

            swidth, sheight = sprite.size
            sprite.x, sprite.y = world_coord_to_screen_coord(
                position.x, position.y, camera)
            sprite.x -= swidth
            sprite.y -= sheight
            

def run():
    global camera, TIME_SCALE

    astronomical_objects = []
    
    sdl2.ext.init()
    window = sdl2.ext.Window("Solar Orbiters", size=WINDOW_SIZE)
    window.show()

    if "-hardware" in sys.argv:
        print("Using hardware acceleration")
        renderer = sdl2.ext.Renderer(window)
        factory = sdl2.ext.SpriteFactory(sdl2.ext.TEXTURE, renderer=renderer)
    else:
        print("Using software rendering")
        factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)

    world = sdl2.ext.World()

    # movement = MovementSystem(0, 0, 800, 600)
    
    if factory.sprite_type == sdl2.ext.SOFTWARE:
        spriterenderer = SoftwareRenderSystem(window)
    else:
        spriterenderer = TextureRenderSystem(renderer)

    movementsystem = MovementSystem()
    spritemovementsystem = SpriteMovementSystem()
    camera = Camera()

    world.add_system(spriterenderer)
    world.add_system(movementsystem)
    world.add_system(spritemovementsystem)

    # Parse solar system data from xml
    tree = ET.parse(ASTRO_OBJECTS_XML_PATH)
    root = tree.getroot()

    # Instantiate planets
    for astro_object in root.findall('object'):
        color_elem = astro_object.find('color')
        color = sdl2.ext.Color(int(color_elem[0].text),
                               int(color_elem[1].text),
                               int(color_elem[2].text))
        sprite = factory.from_color(color, size=(10, 10))
        mass = ( float(astro_object.find('mass').text) *
                 10 ** int(astro_object.find('mass')[0].text))
        x = int(astro_object.find('position').find('x').text) * 1000
        y = int(astro_object.find('position').find('y').text) * 1000
        vx = float(astro_object.find('velocity').find('x').text) * 1000
        vy = float(astro_object.find('velocity').find('y').text) * 1000
        astronomical_objects.append(AstronomicalObject(world, sprite,
                                                       mass, x, y, vx, vy))

    # Instantiate some Trojans... or were they Greeks?
    # Pretty messy. Should clean up a bit.
    for i in xrange(0):
        sprite = factory.from_color(GRAY, size=(4, 4))
        mass = randint(1, 10000000) # Apparently, they're light. ;)
        # Put them on the same orbit as Jupiter.
        origin = 778412010000
        x0 = origin*cos(pi/3)
        y0 = origin*sin(pi/3)
        # Add noise to location.
        radius = randint(0, 100000000000)
        pos_angle = vonmisesvariate(0,0)
        x = int(cos(pos_angle) * radius + x0)
        y = int(sin(pos_angle) * radius + y0)
        # Start with orbital speed identical to that of Jupiter's.
        vel0 = 13.0697 * 1000
        vx0 = vel0 * cos(pi/3+pi/2)
        vy0 = vel0 * sin(pi/3+pi/2)
        # Add significant noise to velocity.
        vel_angle = vonmisesvariate(0,0)
        velocity = uniform(0,200)
        vx = cos(vel_angle) * velocity + vx0
        vy = sin(vel_angle) * velocity + vy0
        astronomical_objects.append(AstronomicalObject(world, sprite,
                                                       mass, x, y, vx, vy))
        

    running = True
    while running:
        for event in sdl2.ext.get_events():
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
            if event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_UP:
                    camera.move(0,-10)
                elif event.key.keysym.sym == sdl2.SDLK_DOWN:
                    camera.move(0,10)
                elif event.key.keysym.sym == sdl2.SDLK_LEFT:
                    camera.move(-10,0)
                elif event.key.keysym.sym == sdl2.SDLK_RIGHT:
                    camera.move(10,0)
                elif event.key.keysym.sym == sdl2.SDLK_x:
                    camera.zoom(1.1111111)
                elif event.key.keysym.sym == sdl2.SDLK_z:
                    camera.zoom(0.9)
                elif event.key.keysym.sym == sdl2.SDLK_PERIOD:
                    TIME_SCALE += 1
                elif event.key.keysym.sym == sdl2.SDLK_COMMA:
                    TIME_SCALE = max(1, TIME_SCALE-1)
        sdl2.SDL_Delay(10)
        apply_gravity(astronomical_objects)
        world.process()


if __name__ == "__main__":
    sys.exit(run())
