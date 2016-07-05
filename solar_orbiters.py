#!/usr/bin/env python3
"""Solar orbiters

A numerical simulation of the planets of the Solar system.
Also, some asteroids.

Requires Python 3 and pysdl2 installed. May require fiddling. (Windows: E.g.
correct SDL2.dll in the System32 directory.)

DEBUG: Weird loss of gravity and disappearance of Jupiter (singularity?)

TODO: Extend to 3d.
    TODO: Extend Camera to 3d.
"""

import sys
import os
#os.environ["PYSDL2_DLL_PATH"] = "c:\\Python27\\DLLs"
import sdl2
import sdl2.ext
from math import *
from random import *
import xml.etree.ElementTree as ET

ASTRO_OBJECTS_XML_PATH = 'astro_objects.xml'
WINDOW_SIZE = 800, 600
WINDOW_SCALE = 150e9 // 50 # 150 exp 9 m = about 1 AU = 50 pixels

BLACK  = sdl2.ext.Color(0, 0, 0)
WHITE  = sdl2.ext.Color(255, 255, 255)
BLUE   = sdl2.ext.Color(0, 0, 255)
YELLOW = sdl2.ext.Color(255, 255, 0)
GRAY   = sdl2.ext.Color(150, 150, 150)

GRAV_CONSTANT = 6.67408e-11 # For meters!

SECONDS_PER_STEP = 3600 # The computation step in seconds
STEPS_PER_FRAME = 0 # Computational steps per frame (supposedly 10 ms per
                    # frame). Can be adjusted with keyboard.
THETA = 0.5 # Distance threshold ratio. Large values increase speed but
            # sacrifice accuracy.
MAX_QUADTREE_DEPTH = 30

TROJANS = 20
FREE_ASTEROIDS = 20
JUPITER_ORBITERS = 20

EXTRA_PLANETOIDS = 0

MAKE_PLANETS = True # Debug option to disable normal planet creation, inc. Sun



class SpriteMovementSystem(sdl2.ext.Applicator):
    """Makes sprites represent the positions of astronomical objects."""
    def __init__(self):
        super(SpriteMovementSystem, self).__init__()
        self.componenttypes = Position, sdl2.ext.Sprite

    def process(self, world, componentsets):
        """Move sprites to represent planet movement"""
        local_camera = camera # Minor hack to minimize global variable access.
        for position, sprite in componentsets:
            swidth, sheight = sprite.size
            sprite.x, sprite.y = local_camera.world_coord_to_screen_coord(
                position.x, position.y)
            sprite.x -= swidth // 2
            sprite.y -= sheight // 2
        


class MovementSystem(sdl2.ext.Applicator):
    """Applies gravity and manages movement of astronomical objects."""
    def __init__(self):
        super(MovementSystem, self).__init__()
        self.componenttypes = (Mass, Position, Velocity, Acceleration)
        self.is_gravity_initialized = False

    def process(self, world, componentsets):
        """Apply Barnes-Hut gravity algorithm (O(n log n))"""
        # Squeeze some efficiency with local variables
        time_step = SECONDS_PER_STEP
        grav_constant = GRAV_CONSTANT
        # Super clunky, but componentsets is apparently an iterator, not a list!
        comps = list(componentsets)
        
        for i in range(STEPS_PER_FRAME):
            grav_data_tuples = [(mass.mass, position.x, position.y, position.z)
                                for mass, position, velocity, acceleration in
                                comps]

            root_node = QuadNode(grav_data_tuples, 10e13, 0, 0, 0, is_root=True)
            
            for mass, position, velocity, acceleration in comps:
                # Compute gravitational acceleration for step
                ax, ay, az = root_node.get_gravity_at_point(position.x,
                                                            position.y,
                                                            position.z)
                acceleration.ax = ax
                acceleration.ay = ay
                acceleration.az = az
                
                velocity.vx += ax * time_step
                velocity.vy += ay * time_step
                velocity.vz += az * time_step

                position.x += velocity.vx * time_step
                position.y += velocity.vy * time_step
                position.z += velocity.vz * time_step
                

class QuadNode():
    """A basic data structure tailored for the Barnes-Hut algorithm."""
    def __init__(self, data_tuples, width, x, y, z, is_root=False, depth=0):
        # data_tuples = [(mass, x, y)...]
        self.width = width
        self.mass = 0
        self.center_of_gravity_x = 0
        self.center_of_gravity_y = 0
        self.center_of_gravity_z = 0
        self.is_internal = True
        self.is_root = is_root
        self.children = []
        self.depth = depth

        if self.is_root:
            x_by_mass = 0
            y_by_mass = 0
            z_by_mass = 0
            total_mass = 0
            
            for o in data_tuples:
                object_mass = o[0]
                total_mass += object_mass
                object_x = o[1]
                object_y = o[2]
                object_z = o[3]
                x_by_mass += object_x * object_mass
                y_by_mass += object_y * object_mass
                z_by_mass += object_z * object_mass

            self.x = x_by_mass / total_mass
            self.y = y_by_mass / total_mass
            self.z = z_by_mass / total_mass
            
        length = len(data_tuples)

        if self.is_root: # Debug...
            print(length)
        
        if length > 1 and self.depth < MAX_QUADTREE_DEPTH:
            lists = ([],[],[],[],[],[],[],[])
            x_by_mass = 0
            y_by_mass = 0
            z_by_mass = 0
            
            for o in data_tuples:
                object_mass = o[0]
                self.mass += object_mass
                object_x, object_y, object_z = o[1:4]
                
                x_by_mass += object_x * object_mass
                y_by_mass += object_y * object_mass
                z_by_mass += object_z * object_mass

                list_index = 0
                if object_x > x:
                    list_index += 4
                if object_y > y:
                    list_index += 2
                if object_z > z:
                    list_index += 1
                lists[list_index].append(o)

            self.center_of_gravity_x = x_by_mass / self.mass
            self.center_of_gravity_y = y_by_mass / self.mass
            self.center_of_gravity_z = z_by_mass / self.mass
            
            halfwidth = width/2

            for i in range(8):
                if lists[i]:
                    node_x = x + halfwidth if i // 4 else x - halfwidth
                    node_y = y + halfwidth if i % 4 // 2 else y - halfwidth
                    node_z = z + halfwidth if i % 2 else z - halfwidth
                    
                    self.children.append(
                        QuadNode(lists[i], halfwidth, node_x, node_y, node_z,
                                 depth=self.depth+1))
                
        elif length > 1 and self.depth == MAX_QUADTREE_DEPTH:
            # Most likely at least two objects are superimposed.
            x_by_mass = 0
            y_by_mass = 0
            z_by_mass = 0
            
            for o in data_tuples:
                object_mass = o[0]
                self.mass += object_mass
                object_x = o[1]
                object_y = o[2]
                object_z = o[3]
                x_by_mass += object_x * object_mass
                y_by_mass += object_y * object_mass
                z_by_mass += object_z * object_mass
                self.children.append(QuadNode([o,], width,
                                              x, y, z,
                                              depth=self.depth+1))

            self.center_of_gravity_x = x_by_mass / self.mass
            self.center_of_gravity_y = y_by_mass / self.mass
            self.center_of_gravity_z = z_by_mass / self.mass
            
        elif length == 1:
            self.is_internal = False
            astro_object = data_tuples[0]
            self.mass = astro_object[0]
            self.center_of_gravity_x = astro_object[1]
            self.center_of_gravity_y = astro_object[2]
            self.center_of_gravity_z = astro_object[3]
            
        else: # Should never happen.
            self.is_internal = False
            self.center_of_gravity = 0, 0, 0
            
    def get_gravity_at_point(self, x, y, z):
        """Calculate the gravity exerted by this node at given point."""
        if self.mass == 0: # Should never happen.
            return 0.0, 0.0, 0.0
        elif self.is_accurate_enough(x, y, z): # Avert singularity
            if self.center_of_gravity_x == x and \
               self.center_of_gravity_y == y and \
               self.center_of_gravity_z == z:
                return 0.0, 0.0, 0.0
            delta_x = self.center_of_gravity_x - x
            delta_y = self.center_of_gravity_y - y
            delta_z = self.center_of_gravity_z - z

            # Gravitational acceleration generated at this location
            # by given object, as per Newton's gravitational equation,
            # but divided on both sides by the mass of the affected
            # object. (Norm.)
            a = GRAV_CONSTANT * self.mass / (delta_x * delta_x +
                                             delta_y * delta_y +
                                             delta_z * delta_z)

            """TODO: Add actual Vector3 class to get directions in a reasonably
            sane fashion."""
            a_unit_vector = Vector3(delta_x, delta_y, delta_z).unit_vector()
            a_vector = a_unit_vector.mul(a)
            return a_vector.x, a_vector.y, a_vector.z
        else:
            ax, ay, az = 0, 0, 0
            for n in self.children:
                child_ax, child_ay, child_az = n.get_gravity_at_point(x, y, z)
                ax += child_ax
                ay += child_ay
                az += child_az
            return ax, ay, az

    def is_accurate_enough(self, x, y, z):
        """Determine if the current node is accurate enough for the given
        point."""
        if self.is_internal:
            delta_x = self.center_of_gravity_x - x
            delta_y = self.center_of_gravity_y - y
            delta_z = self.center_of_gravity_y - z
            distance_squared = (delta_x * delta_x +
                                delta_y * delta_y +
                                delta_z * delta_z)
                                
            if distance_squared == 0: # Avert div by zero
                return False
            elif self.width * self.width / distance_squared <= THETA * THETA:
                return True
            else:
                return False
        else:
            return True


class Vector3:
    """A minimalist 3-dimensional vector."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def abs(self):
        """Return vector magnitude."""
        return sqrt(self.abs_squared())

    def abs_squared(self):
        """Return vector magnitude squared (to avoid expensive sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def unit_vector(self):
        """Return unit vector."""
        norm = self.abs()
        x = self.x / norm
        y = self.y / norm
        z = self.z / norm
        return Vector3(x, y, z)

    def mul(self, other):
        return Vector3(self.x*other, self.y*other, self.z*other)
    


class SoftwareRenderSystem(sdl2.ext.SoftwareSpriteRenderSystem):
    """The default renderer."""
    def __init__(self, window):
        super(SoftwareRenderSystem, self).__init__(window)

    def render(self, components):
        sdl2.ext.fill(self.surface, BLACK)
        super(SoftwareRenderSystem, self).render(components)


class TextureRenderSystem(sdl2.ext.TextureSpriteRenderSystem):
    """Hardware-accelerated renderer. Not default."""
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
    """A simple class to transform world co-ordinates to screen co-ordinates.
    Allows movement of display."""
    def __init__(self):
        self.x = 0
        self.y = 0
        self.scale = WINDOW_SCALE

    def move(self, dx, dy):
        self.x += dx * self.scale
        self.y += dy * self.scale

    def zoom(self, factor):
        self.scale = int(self.scale / factor)

    def world_coord_to_screen_coord(self, x_world, y_world):
        w_width, w_height = WINDOW_SIZE
        x_screen = int((x_world - self.x) / self.scale + w_width / 2 )
        y_screen = int((y_world - self.y) / self.scale + w_height / 2 )
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
        self.z = 0

class Velocity(object):
    def __init__(self):
        super(Velocity, self).__init__()
        self.vx = 0
        self.vy = 0
        self.vz = 0

class Acceleration(object):
    def __init__(self):
        super(Acceleration, self).__init__()
        self.ax = 0
        self.ay = 0
        self.az = 0
        

class AstronomicalObject(sdl2.ext.Entity):
    """Model of an astronomical object (eg. star, planet, moon, asteroid)."""
    def __init__(self, world, sprite, mass=0, posx=0, posy=0, posz=0,
                 vx=0, vy=0, vz=0):
        self.sprite = sprite
        self.sprite.position = camera.world_coord_to_screen_coord(posx,posy)

        self.position = Position()
        self.position.x = posx
        self.position.y = posy
        self.position.z = posz

        self.velocity = Velocity()
        self.velocity.vx = vx
        self.velocity.vy = vy
        self.velocity.vz = vz
        
        self.acceleration = Acceleration()
        self.mass = Mass()
        self.mass.mass = mass
            

def run():
    global camera, STEPS_PER_FRAME

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
        if not MAKE_PLANETS:
            break
        color_elem = astro_object.find('color')
        color = sdl2.ext.Color(int(color_elem[0].text),
                               int(color_elem[1].text),
                               int(color_elem[2].text))
        sprite = factory.from_color(color, size=(10, 10))
        mass = ( float(astro_object.find('mass').text) *
                 10 ** int(astro_object.find('mass')[0].text))
        x = int(astro_object.find('position').find('x').text) * 1000
        y = int(astro_object.find('position').find('y').text) * 1000
        z = int(astro_object.find('position').find('z').text) * 1000
        vx = float(astro_object.find('velocity').find('x').text) * 1000
        vy = float(astro_object.find('velocity').find('y').text) * 1000
        vz = float(astro_object.find('velocity').find('z').text) * 1000
        astronomical_objects.append(AstronomicalObject(world, sprite, mass,
                                                       x, y, z,  vx, vy, vz))

    # TODO: Refactor asteroid creation to function.
    # Instantiate some Trojans... or were they Greeks?
    # Pretty messy. Should clean up a bit.
    for i in range(TROJANS):
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
        z = 0
        # Start with orbital speed identical to that of Jupiter's.
        vel0 = 13.0697 * 1000
        vx0 = vel0 * cos(pi/3+pi/2)
        vy0 = vel0 * sin(pi/3+pi/2)
        # Add significant noise to velocity.
        vel_angle = vonmisesvariate(0,0)
        velocity = uniform(0,200)
        vx = cos(vel_angle) * velocity + vx0
        vy = sin(vel_angle) * velocity + vy0
        vz = 0
        astronomical_objects.append(AstronomicalObject(world, sprite, mass,
                                                       x, y, z, vx, vy, vz))

    # Instantiate some Jupiter Orbiters
    # Pretty messy. Should clean up a bit.
    for i in range(JUPITER_ORBITERS):
        sprite = factory.from_color(GRAY, size=(4, 4))
        mass = randint(1, 10000000) # Apparently, they're light. ;)
        # Put them on the same orbit as Jupiter.
        x0, y0 = 778412010000, 0
        # Add noise to location.
        radius = randint(1e3, 1e11)
        pos_angle = vonmisesvariate(0,0)
        x = int(cos(pos_angle) * radius + x0)
        y = int(sin(pos_angle) * radius + y0)
        z = 0
        # Start with orbital speed identical to that of Jupiter's.
        vx0, vy0 = 0, 13.0697 * 1000
        # Add significant noise to velocity.
        vel_angle = vonmisesvariate(0,0)
        velocity = uniform(0,1e3)
        vx = cos(vel_angle) * velocity + vx0
        vy = sin(vel_angle) * velocity + vy0
        vz = 0
        astronomical_objects.append(AstronomicalObject(world, sprite, mass,
                                                       x, y, z, vx, vy, vz))

    # Instantiate some random asteroids.
    # Pretty messy. Should clean up a bit.
    for i in range(FREE_ASTEROIDS):
        sprite = factory.from_color(GRAY, size=(4, 4))
        mass = randint(1, 10000000) # Apparently, they're light. ;)
        # Put them on the same orbit as Jupiter.
        x0, y0 = 0, 0
        # Add noise to location.
        radius = randint(1e5, 1e12)
        pos_angle = vonmisesvariate(0,0)
        x = int(cos(pos_angle) * radius + x0)
        y = int(sin(pos_angle) * radius + y0)
        z = 0
        # Add significant noise to velocity.
        vel_angle = vonmisesvariate(0,0)
        velocity = uniform(0,1e5)
        vx = cos(vel_angle) * velocity
        vy = sin(vel_angle) * velocity
        vz = 0
        astronomical_objects.append(AstronomicalObject(world, sprite, mass,
                                                       x, y, z, vx, vy, vz))

    # Instantiate some random planetoids.
    # Pretty messy. Should clean up a bit.
    for i in range(EXTRA_PLANETOIDS):
        sprite = factory.from_color(GRAY, size=(10, 10))
        mass = 10e26 # Boring, heavy
        # Put them dead center.
        x0, y0 = 0, 0
        # Add noise to location.
        radius = randint(1e5, 1e12)
        pos_angle = vonmisesvariate(0,0)
        x = int(cos(pos_angle) * radius + x0)
        y = int(sin(pos_angle) * radius + y0)
        z = 0
        # Add significant noise to velocity.
        vel_angle = vonmisesvariate(0,0)
        velocity = uniform(0,1e5)
        vx = cos(vel_angle) * velocity
        vy = sin(vel_angle) * velocity
        vz = 0
        astronomical_objects.append(AstronomicalObject(world, sprite, mass,
                                                       x, y, z, vx, vy, vz))
        
    
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
                    STEPS_PER_FRAME += 1
                elif event.key.keysym.sym == sdl2.SDLK_COMMA:
                    STEPS_PER_FRAME = max(0, STEPS_PER_FRAME-1)
                    
        sdl2.SDL_Delay(10)
        
        world.process()


if __name__ == "__main__":
    sys.exit(run())
