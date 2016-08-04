#!/usr/bin/env python3
"""Solar orbiters

A numerical, Newtonian and very inaccurate simulation of the planets of the
Solar system. Also, some asteroids.

Requires Python 3 and pysdl2 installed. May require fiddling. (Windows: E.g.
correct SDL2.dll in the System32 directory.)

Keys:
    w / s: pitch down/up
    a / d: yaw left/right
    q / r: roll left/right
    x / z: zoom in/out
    h / n: move forward/backward
    i / k: move up/down
    j / l: move left, right
    , / .: increase/decrease simulation speed (gets choppy fast) 

TODO: Move redundant code (or code of very general nature) somewhere else?
    TODO: Move generic Octree and Octnode to their own module.
    TODO: Move Vector3 to it's own module.

TODO: Improve graphical output.
    TODO: Draw current velocity as a vector
    TODO: Draw gravitational acceleration as a vector
    TODO: Draw orbits (raw estimate from velocity and acceleration)

TODO: Improve Vector3 class
    TODO: Make immutable.
    TODO: Override operators.

TODO: Clean up asteroid generation.

TODO: Migrate from xml to json.

TODO: Get real-world data for planet positions and velocities.
    From NASA Horizons?
"""

import sys
import os
import sdl2
import sdl2.ext
import math
import random
import xml.etree.ElementTree as ET
from typing import Iterable
import quaternion as quat

ASTRO_OBJECTS_XML_PATH = 'astro_objects.xml'
WINDOW_SIZE = 800, 600
WINDOW_SCALE = 150e9 // 50 # 150 exp 9 m = about 1 AU = 50 pixels

BLACK  = sdl2.ext.Color(0, 0, 0)
WHITE  = sdl2.ext.Color(255, 255, 255)
BLUE   = sdl2.ext.Color(0, 0, 255)
YELLOW = sdl2.ext.Color(255, 255, 0)
GRAY   = sdl2.ext.Color(150, 150, 150)

GRAV_CONSTANT = 6.67408e-11 # For meters!

SECONDS_PER_STEP = 3600 # The computation step in seconds. Rather high.
STEPS_PER_FRAME = 0 # Computational steps per frame (supposedly 10 ms per
                    # frame). Can be adjusted with keyboard.
THETA = 0.5 # Distance threshold ratio. Large values increase speed but
            # sacrifice accuracy.
MAX_QUADTREE_DEPTH = 30

NUM_OF_STARS = 1001

# Number of extra objects to twirl in the simulation.
TROJANS = 0
FREE_ASTEROIDS = 0
JUPITER_ORBITERS = 0
EXTRA_PLANETOIDS = 0

MAKE_PLANETS = True # Debug option to disable normal planet creation, inc. Sun



class SpriteMovementSystem(sdl2.ext.Applicator):
    """Makes sprites represent the positions of astronomical objects."""
    def __init__(self):
        super(SpriteMovementSystem, self).__init__()
        self.componenttypes = Position, sdl2.ext.Sprite

    def process(self, world, componentsets):
        """Move sprites to represent planet movement"""
        global camera
        
        for position, sprite in componentsets:
            sprite.x, sprite.y, sprite.depth = (
                camera.world_to_screen_space(position))


class MovementSystem(sdl2.ext.Applicator):
    """Applies gravity and manages movement of astronomical objects."""
    def __init__(self):
        super(MovementSystem, self).__init__()
        self.componenttypes = (Mass, Position, Velocity, Acceleration)
        self.is_gravity_initialized = False

    def process(self, world, componentsets):
        # Squeeze some efficiency with local variables
        time_step = SECONDS_PER_STEP
        grav_constant = GRAV_CONSTANT
        ts_squared = time_step * time_step
        # Super clunky, but componentsets is apparently an iterator, not a list!
        comps = list(componentsets)
        for i in range(STEPS_PER_FRAME):
            self._apply_gravity(time_step, ts_squared, grav_constant, comps)
            self._move_objects(time_step, comps)
            #self._check_collisions(time_step, comps)

    def _apply_gravity(self, time_step, ts_squared, grav_constant, comps):
        """Apply Barnes-Hut gravity algorithm (O(n log n))"""
        grav_data_tuples = [(mass.mass, position)
                            for mass, position, velocity, acceleration in
                            comps]

        gravity_tree = BarnesHutOctree(grav_data_tuples)

        for mass, position, velocity, acceleration in comps:
            # Compute gravitational acceleration for step
            gravity = gravity_tree.get_gravity(position)
            acceleration.x = gravity.x
            acceleration.y = gravity.y
            acceleration.z = gravity.z

    def _move_objects(self, time_step, comps):
        ts_squared = time_step * time_step
        for mass, position, velocity, acceleration in comps:
            ax = acceleration.x
            ay = acceleration.y
            az = acceleration.z
            

            position.x += (velocity.x * time_step +
                           0.5 * ax * ts_squared)
            position.y += (velocity.y * time_step +
                           0.5 * ay * ts_squared)
            position.z += (velocity.z * time_step +
                           0.5 * az * ts_squared)
            
            velocity.x += ax * time_step
            velocity.y += ay * time_step
            velocity.z += az * time_step

    def _check_collisions(self, time_steps, comps):
        collision_distance = 1e6 # Magic number: 1000 km, say
        position_list = [Vector3(position.x, position.y, position.z)
                         for mass, position, velocity, acceleration in
                         comps]
        position_octree = Octree(position_list)
        for v in position_list:
            v_min = v.sub(collision_distance)
            v_max = v.add(collision_distance)
            colliders = position_octree.objects_within_bb(v_min, v_max)
            if len(colliders) > 1: # Just a debug print atm.
                print('Collision! {}'.format(v))
                print(position_octree)
                for pos in position_list:
                    print(pos)
            
            
class Octree():
    """A basic octree data structure.

    For current state, data_list should be a list of Vector3."""
    def __init__(self, data_list, max_obj_per_leaf=3, max_depth=30):
        self.root = OctNode(data_list, max_obj_per_leaf, 0, max_depth)
        

    def objects_within_bb(self, position_min, position_max, node=None,
                          objects=None):
        """Return all objects within bounding box defined by two 3-vectors,
        position_min and position_max. Shouldn't cause infinite recursion."""
        if node is None:
            node = self.root
        if objects is None:
            objects = []

        if node.is_leaf:
            for o in node.children:
                if o.strictly_ge(position_min) and \
                   o.strictly_le(position_max):
                    objects.append(o)
        else:
            for n in node.children:
                if n is None:
                    continue
                if n.min.strictly_le(position_max) and \
                   n.max.strictly_ge(position_min):
                    self.objects_within_bb(position_min, position_max, node=n,
                                           objects=objects)
        return objects

    def __str__(self):
        string = 'Octree, nodes:\n'
        string += str(self.root) # Recursive, mind you!
        return string


class OctNode():
    """A helper class for Octree"""
    def __init__(self, data_list, max_obj_per_leaf, depth, max_depth, path=''):
        self.is_leaf = True
        self.children = []
        self.depth = depth
        self.min = None
        self.max = None
        self.center = None
        self.path = path
        self._set_bbox(data_list)
        self._fill_node(data_list, max_obj_per_leaf, max_depth)

    def __str__(self):
        string = self.path + ' OctNode\n'
        if self.is_leaf:
            vector_strings = [' ' * self.depth + str(c) + '\n' for c in
                              self.children]
            string += ''.join(vector_strings)
        else:
            node_strings = [str(node) for node in self.children]
            string += ''.join(node_strings)
        return string
            

    def _set_bbox(self, data_list):
        """Set the bounding box and center for the node."""
        if not data_list:
            return
        self.min = data_list[0].add(0)
        self.max = self.min.add(0)
        for v in data_list:
            if v.x < self.min.x:
                   self.min.x = v.x
            elif v.x > self.max.x:
                   self.max.x = v.x
            if v.y < self.min.y:
               self.min.y = v.y
            elif v.y > self.max.y:
                self.max.y = v.y
            if v.z < self.min.z:
                self.min.z = v.z
            elif v.z > self.max.z:
                self.max.z = v.z
        # Ugly below. Override operators in Vector3. Also a bit superfluous.
        self.center = self.min.add(self.max.sub(self.min).mul(0.5))

    def _fill_node(self, data_list, max_obj_per_leaf, max_depth):
        """Add children to node, unless more data than limit, in which case
        create new nodes as children and divide the data among them."""
        if len(data_list) < max_obj_per_leaf or self.depth >= max_depth:
            self.children = data_list
        else:
            self.is_leaf = False
            bins = ([],[],[],[],[],[],[],[])
            for o in data_list:
                index = 0
                if o.x > self.center.x:
                    index += 4
                if o.y > self.center.y:
                    index += 2
                if o.z > self.center.z:
                    index += 1
                bins[index].append(o)
            for i in range(len(bins)):
                if len(bins[i]) > 0:
                    self.children.append(OctNode(bins[i], max_obj_per_leaf,
                                                 self.depth+1, max_depth,
                                                 path=self.path+str(i)))
                
class BarnesHutOctree:
    """A data structure tailored for the Barnes-Hut algorithm.

    See: https://en.wikipedia.org/wiki/Barnes-Hut_simulation

    TODO: Refine bin division conditions.
    TODO: Consider moving code from node class to tree class."""
    def __init__(self, data_tuples, max_obj_per_leaf=4, max_depth=30,
                 min_width=1000):
        # data_tuples : [(mass, position_vector), ...]
        self.root = BarnesHutOctNode2(data_tuples, max_obj_per_leaf, 0,
                                      max_depth, min_width)
        self.root.init_gravity()

    def get_gravity(self, position):
        return self.root.get_gravity(position)
    

class BarnesHutOctNode2:
    def __init__(self, data_tuples, max_obj_per_leaf, depth, max_depth,
                 min_width):
        self.is_leaf = True
        self.children = []
        self.depth = depth
        self.min = None
        self.max = None
        self.center = None
        self.width = None
        self.center_of_gravity = None
        self.mass = 0
        self._set_bbox(dt[1] for dt in data_tuples)
        self._fill_node(data_tuples, max_obj_per_leaf, max_depth, min_width)
        

    def _set_bbox(self, data_iterable):
        """Set the bounding box and center for the node."""
        first_v = next(data_iterable)
        x_min, y_min, z_min = first_v.x, first_v.y, first_v.z
        x_max, y_max, z_max = first_v.x, first_v.y, first_v.z
        for v in data_iterable:
            if v.x < x_min:
                   x_min = v.x
            elif v.x > x_max:
                   x_max = v.x
            if v.y < y_min:
               y_min = v.y
            elif v.y > y_max:
                y_max = v.y
            if v.z < z_min:
                z_min = v.z
            elif v.z > z_max:
                z_max = v.z
        self.min = Vector3(x_min, y_min, z_min)
        self.max = Vector3(x_max, y_max, z_max)
        # Ugly below. Override operators in Vector3. Also a bit superfluous.
        self.center = self.min.add(self.max.sub(self.min).mul(0.5))
        self.width = self.max.sub(self.min).abs()

    def _fill_node(self, data_list, max_obj_per_leaf, max_depth, min_width):
        """Add children to node. If more data than limit, create new nodes."""
        if len(data_list) < max_obj_per_leaf or self.depth >= max_depth or \
           self.width < min_width:
            self.children = data_list
        else:
            self.is_leaf = False
            bins = ([],[],[],[],[],[],[],[])
            for elem in data_list:
                index = 0
                if elem[1].x > self.center.x:
                    index += 4
                if elem[1].y > self.center.y:
                    index += 2
                if elem[1].z > self.center.z:
                    index += 1
                bins[index].append(elem)
            for i in range(len(bins)):
                if len(bins[i]) > 0:
                    self.children.append(
                        BarnesHutOctNode2(bins[i], max_obj_per_leaf,
                                          self.depth+1, max_depth, min_width))

    def init_gravity(self):
        """Initialize gravity in the node and its children."""
        position_mass_product_sum = Vector3()
        mass_sum = 0

        if self.is_leaf:
            for mass, position in self.children:
                mass_sum += mass
                position_mass_product_sum = position_mass_product_sum.add(
                    position.mul(mass))
        else:
            for node in self.children:
                node.init_gravity()
                mass_sum += node.mass
                position_mass_product_sum = position_mass_product_sum.add(
                    node.center_of_gravity.mul(node.mass))

        self.mass = mass_sum
        self.center_of_gravity = position_mass_product_sum.truediv(mass_sum)

    def get_gravity(self, position):
        """Return gravitational acceleration exerted by this node (or its
        children) at the given position."""
        gravity = Vector3()
        
        if self.is_leaf:
            for mass, mass_position in self.children:
                partial_gravity = self.calculate_gravity(mass, mass_position,
                                                         position)
                gravity = gravity.add(partial_gravity)
        elif self._is_accurate_enough(position):
            gravity = self.calculate_gravity(self.mass, self.center_of_gravity,
                                            position)
        else:
            for node in self.children:
                partial_gravity = node.get_gravity(position)
                gravity = gravity.add(partial_gravity)
                
        return gravity
                
    def calculate_gravity(self, mass, mass_position, grav_position):
        """Calculate and return gravitational acceleration exerted by the given
        mass at the given position."""
        vector_to_mass = mass_position.sub(grav_position)
        if vector_to_mass.x == 0 and vector_to_mass.y == 0 and \
           vector_to_mass.z == 0:
            # Avoid zero division errors or attempts to exert gravity on self.
            return Vector3()

        gravity_mag = GRAV_CONSTANT \
                      * mass \
                      / vector_to_mass.abs_squared()
        gravity = vector_to_mass.unit_vector().mul(gravity_mag)
        return gravity


    def _is_accurate_enough(self, position):
        """Determine if current node is accurate enough for gravity
        approximation."""
        vector_to_position = position.sub(self.center_of_gravity)
        distance_squared = vector_to_position.abs_squared() # Avoid sqrt.
        if distance_squared == 0:
            return True
        return self.width * self.width / distance_squared <= THETA * THETA


class Vector3:
    """A minimalist 3-dimensional vector.

    TODO: Override operators."""
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return 'Vector3: [{}, {}, {}]'.format(self.x, self.y, self.z)

    def abs(self):
        """Return vector magnitude."""
        return math.sqrt(self.abs_squared())

    def abs_squared(self):
        """Return vector magnitude squared (to avoid expensive sqrt)."""
        return self.dot(self)

    def unit_vector(self):
        """Return unit vector."""
        norm = self.abs()
        x = self.x / norm
        y = self.y / norm
        z = self.z / norm
        return Vector3(x, y, z)

    def mul(self, other):
        """Return product of vector and scalar."""
        return Vector3(self.x * other, self.y * other, self.z * other)

    def dot(self, other):
        """Return dot product of vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """Return cross product of vectors.
        TODO: Implement. Requires quaternions,
        so import my-little-quaternion."""
        return None

    def add(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x+other.x, self.y+other.y, self.z+other.z)
        elif isinstance(other, (int, float)):
            # Not mathematically correct, but could be handy?
            return Vector3(self.x+other, self.y+other, self.z+other)
        else:
            raise TypeError('Unsupported type.')

    def sub(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x-other.x, self.y-other.y, self.z-other.z)
        elif isinstance(other, (int, float)):
            # Not mathematically correct, but could be handy?
            return Vector3(self.x-other, self.y-other, self.z-other)
        else:
            raise TypeError('Unsupported type.')

    def truediv(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError('Only int and float supported.')
        return Vector3(self.x/other, self.y/other, self.z/other)

    def floordiv(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError('Only int and float supported.')
        return Vector3(self.x//other, self.y//other, self.z//other)

    def strictly_le(self, other):
        if not isinstance(other, Vector3):
            raise TypeError('Only Vector3 supported.')
        if self.x <= other.x and self.y <= other.y and self.z <= other.z:
            return True
        else:
            return False

    def strictly_ge(self, other):
        if not isinstance(other, Vector3):
            raise TypeError('Only Vector3 supported.')
        if self.x >= other.x and self.y >= other.y and self.z >= other.z:
            return True
        else:
            return False

    def eq(self, other):
        if not isinstance(other, Vector3):
            raise TypeError('Only Vector3 supported.')
        if self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        else:
            return False
        


class SoftwareRenderSystem(sdl2.ext.SoftwareSpriteRenderSystem):
    """Software renderer. Not default."""
    def __init__(self, window):
        super(SoftwareRenderSystem, self).__init__(window)

    def render(self, components):
        sdl2.ext.fill(self.surface, BLACK)
        super(SoftwareRenderSystem, self).render(components)


class TextureRenderSystem(sdl2.ext.TextureSpriteRenderSystem):
    """Hardware-accelerated renderer. Default."""
    def __init__(self, renderer):
        super(TextureRenderSystem, self).__init__(renderer)
        self.renderer = renderer

    def render(self, components):
        global camera, stars

        # Draw stars
        tmp = self.renderer.color
        self.renderer.color = BLACK
        self.renderer.clear()
        self.renderer.color = WHITE
        
        star_coords = [] # stars is a global list of Star objects... for now.
        for s in stars:
            coords = camera.star_to_screen_space(s.direction)
            star_coords.extend(coords)
        self.renderer.draw_point(star_coords)
            
        self.renderer.color = tmp
        # Draw sprites
        r = sdl2.SDL_Rect(0, 0, 0, 0)

        rcopy = sdl2.SDL_RenderCopy
        renderer = self.sdlrenderer
        for sp in components:
            if sp.depth >= 0:
                r.x, r.y, r.w, r.h = -1, -1, 1, 1
            else:
                r.w = int(max(1, (sp.size[0] / (-sp.depth / 15e9))))
                r.h = int(max(1, sp.size[1] / (-sp.depth / 15e9)))
                r.x = sp.x - r.w // 2
                r.y = sp.y - r.h // 2
            if rcopy(renderer, sp.texture, None, r) == -1:
                raise SDLError()
        sdl2.SDL_RenderPresent(self.sdlrenderer)


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

class Camera3D:
    """An experimental camera class to display 3d views of the solar system.

    Error creep is an issue: Directional vectors might not always remain
    perpendicular to each other, screwing up the display."""
    def __init__(self, position=None, forward=None, up=None):
        if position is None:
            self.position = quat.Quaternion(0,0,0,3e11) # High above the plane
        else:
            self.position = position
        if forward is None:
            self.forward = quat.Quaternion(0,0,0,-1)
        else:
            self.forward = forward
        if up is None:
            self.up = quat.Quaternion(0,0,1,0)
        else:
            self.up = up
        # Refactor the crap below.
        self.x_min = 0
        self.y_min = 0
        self.x_max, self.y_max = WINDOW_SIZE
        self.x_center = self.x_max // 2
        self.y_center = self.y_max // 2
        self.z = self.x_center # Try this for size.
        self.minimum_distance = 10000 # 10 km

    def pitch(self, phi):
        """Pitch camera, positive phi down."""
        left = (self.up @ self.forward).normalize()
        self.forward = quat.rotate(self.forward, left, phi).normalize()
        self.up = quat.rotate(self.up, left, phi).normalize()

    def yaw(self, phi):
        """Yaw camera, positive phi right."""
        self.forward = quat.rotate(self.forward, self.up, -phi).normalize()

    def roll(self, phi):
        """Roll camera, positive phi right."""
        self.up = quat.rotate(self.up, self.forward, phi).normalize()

    def world_to_screen_space(self, position):
        """Translate given position to screen space."""
        relat_pos = quat.Quaternion(0,
                                    position.x - self.position.x,
                                    position.y - self.position.y,
                                    position.z - self.position.z)
        distance = abs(relat_pos)
        forward_length = relat_pos * self.forward
        if forward_length < self.minimum_distance:
            return -1, -1, 1
        
        vertical_length = relat_pos * self.up
        vertical_tangent = vertical_length / forward_length
        lateral_length = relat_pos * (self.forward @ self.up)
        lateral_tangent = lateral_length / forward_length

        x = self.x_center + self.z * lateral_tangent
        y = self.y_center - self.z * vertical_tangent
        
        return int(x), int(y), -int(distance)

    def star_to_screen_space(self, position):
        """Translate given position to screen space."""
        relat_pos = quat.Quaternion(0,
                                    position.x,
                                    position.y,
                                    position.z)

        forward_length = relat_pos * self.forward
        if forward_length <= 0.0:
            return -1, -1
        
        vertical_length = relat_pos * self.up
        vertical_tangent = vertical_length / forward_length
        lateral_length = relat_pos * (self.forward @ self.up)
        lateral_tangent = lateral_length / forward_length

        x = self.x_center + self.z * lateral_tangent
        y = self.y_center - self.z * vertical_tangent
        
        return int(x), int(y)

    def translate(self, forward, up, right):
        delta_forward = forward * 1e10 * self.forward 
        delta_up = up * 1e10 * self.up
        delta_right = right * 1e10 * (self.forward @ self.up)
        self.position += delta_forward
        self.position += delta_up
        self.position += delta_right

    def zoom(self, increment):
        new_z = max(1, int(self.z * 2 ** increment))
        if new_z == self.z and increment > 0:
            new_z += 1
        self.z = new_z


# Define data bags
class Mass:
    def __init__(self):
        self.mass = 0

class Radius:
    def __init__(self):
        self.radius = 0

class Position(Vector3):
    def __init__(self, x=0, y=0, z=0):
        super(Position, self).__init__(x, y, z)

class Velocity(Vector3):
    def __init__(self, x=0, y=0, z=0):
        super(Velocity, self).__init__(x, y, z)

class Acceleration(Vector3):
    def __init__(self, x=0, y=0, z=0):
        super(Acceleration, self).__init__(x, y, z)

class Direction(Vector3):
    def __init__(self, x=1, y=0, z=0):
        super(Direction, self).__init__(x, y, z)
        

class AstronomicalObject(sdl2.ext.Entity):
    """Model of an astronomical object (eg. star, planet, moon, asteroid)."""
    def __init__(self, world, sprite, mass=0, radius=0, posx=0, posy=0, posz=0,
                 vx=0, vy=0, vz=0):
        self.position = Position(posx, posy, posz)

        self.velocity = Velocity(vx, vy, vz)
        
        self.acceleration = Acceleration()
        self.mass = Mass()
        self.mass.mass = mass

        self.radius = Radius()
        self.radius.radius = radius

        self.sprite = sprite
        self.sprite.position = camera.world_to_screen_space(self.position)


class Star(sdl2.ext.Entity):
    """A cosmetic little dot far, far away."""
    def __init__(self, world, x=1, y=0, z=0):
        self.direction = Direction(x, y, z)
            

def generate_random_directions(number):
    directions = []
    for i in range(number):
        directions.append(spherical_distribution())
    return directions

def spherical_distribution():
    """Generate elements of pseudo-random unit vector with uniform spherical
    distribution.
    See http://math.stackexchange.com/q/44691 (version: 2011-06-11)"""
    theta = random.uniform(0, math.pi*2)
    z = random.uniform(-1.0, 1.0)
    x = math.cos(theta) * math.sqrt(1.0 - z * z)
    y = math.sin(theta) * math.sqrt(1.0 - z * z)

    return x, y, z

def run():
    global camera, STEPS_PER_FRAME, world, stars

    astronomical_objects = []
    stars = []
    
    sdl2.ext.init()
    window = sdl2.ext.Window("Solar Orbiters", size=WINDOW_SIZE)
    window.show()

    if True:
        print("Using hardware acceleration")
        renderer = sdl2.ext.Renderer(window)
        factory = sdl2.ext.SpriteFactory(sdl2.ext.TEXTURE, renderer=renderer)
    else:
        print("Using software rendering")
        factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)

    world = sdl2.ext.World()
    
    if factory.sprite_type == sdl2.ext.SOFTWARE:
        spriterenderer = SoftwareRenderSystem(window)
    else:
        spriterenderer = TextureRenderSystem(renderer)

    movementsystem = MovementSystem()
    spritemovementsystem = SpriteMovementSystem()
    camera = Camera3D()

    world.add_system(spriterenderer)
    world.add_system(movementsystem)
    world.add_system(spritemovementsystem)

    # Instantiate stars
    for x, y, z in generate_random_directions(NUM_OF_STARS):
        stars.append(Star(world, x, y, z))

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
        diameter = max(1, int(float(astro_object.find('diameter').text) * 10))
        sprite = factory.from_color(color, size=(diameter, diameter))
        mass = ( float(astro_object.find('mass').text) *
                 10 ** int(astro_object.find('mass')[0].text))
        radius = 10000 # Temporary test number.
        
        x = int(astro_object.find('position').find('x').text) * 1000
        y = int(astro_object.find('position').find('y').text) * 1000
        z = int(astro_object.find('position').find('z').text) * 1000
        vx = float(astro_object.find('velocity').find('x').text) * 1000
        vy = float(astro_object.find('velocity').find('y').text) * 1000
        vz = float(astro_object.find('velocity').find('z').text) * 1000
        astronomical_objects.append(AstronomicalObject(world, sprite,
                                                       mass, radius,
                                                       x, y, z,  vx, vy, vz))
    
    # TODO: Refactor asteroid creation to function.
    # Instantiate some Trojans... or were they Greeks?
    # Pretty messy. Should clean up a bit.
    for i in range(TROJANS):
        sprite = factory.from_color(GRAY, size=(4, 4))
        mass = random.randint(1, 10000000) # Apparently, they're light. ;)
        # Put them on the same orbit as Jupiter.
        origin = 778412010000
        x0 = origin*math.cos(math.pi/3)
        y0 = origin*math.sin(math.pi/3)
        # Add noise to location.
        radius = random.randint(0, 100000000000)
        pos_angle = random.vonmisesvariate(0,0)
        x = int(math.cos(pos_angle) * radius + x0)
        y = int(math.sin(pos_angle) * radius + y0)
        z = 0
        # Start with orbital speed identical to that of Jupiter's.
        vel0 = 13.0697 * 1000
        vx0 = vel0 * math.cos(math.pi/3+math.pi/2)
        vy0 = vel0 * math.sin(math.pi/3+math.pi/2)
        # Add significant noise to velocity.
        vel_angle = random.vonmisesvariate(0,0)
        velocity = random.uniform(0,200)
        vx = math.cos(vel_angle) * velocity + vx0
        vy = math.sin(vel_angle) * velocity + vy0
        vz = 0
        astronomical_objects.append(AstronomicalObject(world, sprite, mass,
                                                       x, y, z, vx, vy, vz))

    # Instantiate some Jupiter Orbiters
    # Pretty messy. Should clean up a bit.
    for i in range(JUPITER_ORBITERS):
        sprite = factory.from_color(GRAY, size=(4, 4))
        mass = random.randint(1, 10000000) # Apparently, they're light. ;)
        # Put them on the same orbit as Jupiter.
        x0, y0 = 778412010000, 0
        # Add noise to location.
        radius = random.randint(1e3, 1e11)
        pos_angle = random.vonmisesvariate(0,0)
        x = int(math.cos(pos_angle) * radius + x0)
        y = int(math.sin(pos_angle) * radius + y0)
        z = 0
        # Start with orbital speed identical to that of Jupiter's.
        vx0, vy0 = 0, 13.0697 * 1000
        # Add significant noise to velocity.
        vel_angle = random.vonmisesvariate(0,0)
        velocity = random.uniform(0,1e3)
        vx = math.cos(vel_angle) * velocity + vx0
        vy = math.sin(vel_angle) * velocity + vy0
        vz = 0
        astronomical_objects.append(AstronomicalObject(world, sprite, mass,
                                                       x, y, z, vx, vy, vz))

    # Instantiate some random asteroids.
    # Pretty messy. Should clean up a bit.
    for i in range(FREE_ASTEROIDS):
        sprite = factory.from_color(GRAY, size=(4, 4))
        mass = random.randint(1, 10000000) # Apparently, they're light. ;)
        # Put them on the same orbit as Jupiter.
        x0, y0 = 0, 0
        # Add noise to location.
        radius = random.randint(1e5, 1e12)
        pos_angle = random.vonmisesvariate(0,0)
        x = int(math.cos(pos_angle) * radius + x0)
        y = int(math.sin(pos_angle) * radius + y0)
        z = 0
        # Add significant noise to velocity.
        vel_angle = random.vonmisesvariate(0,0)
        velocity = random.uniform(0,1e5)
        vx = math.cos(vel_angle) * velocity
        vy = math.sin(vel_angle) * velocity
        vz = 0
        astronomical_objects.append(AstronomicalObject(world, sprite, mass,
                                                       x, y, z, vx, vy, vz))

    # Instantiate some random planetoids.
    # Pretty messy. Should clean up a bit.
    for i in range(EXTRA_PLANETOIDS):
        sprite = factory.from_color(GRAY, size=(10, 10))
        mass = 1e28 # Boring, heavy
        # Put them dead center.
        x0, y0 = 0, 0
        # Add noise to location.
        radius = random.randint(1e5, 1e12)
        pos_angle = random.vonmisesvariate(0,0)
        x = int(math.cos(pos_angle) * radius + x0)
        y = int(math.sin(pos_angle) * radius + y0)
        z = 0
        # Add significant noise to velocity.
        vel_angle = random.vonmisesvariate(0,0)
        velocity = random.uniform(0,5e4)
        vx = math.cos(vel_angle) * velocity
        vy = math.sin(vel_angle) * velocity
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
                if event.key.keysym.sym == sdl2.SDLK_h:
                    camera.translate(1,0,0)
                elif event.key.keysym.sym == sdl2.SDLK_n:
                    camera.translate(-1,0,0)
                elif event.key.keysym.sym == sdl2.SDLK_i:
                    camera.translate(0,1,0)
                elif event.key.keysym.sym == sdl2.SDLK_k:
                    camera.translate(0,-1,0)
                elif event.key.keysym.sym == sdl2.SDLK_j:
                    camera.translate(0,0,-1)
                elif event.key.keysym.sym == sdl2.SDLK_l:
                    camera.translate(0,0,1)
                elif event.key.keysym.sym == sdl2.SDLK_x:
                    camera.zoom(0.1)
                elif event.key.keysym.sym == sdl2.SDLK_z:
                    camera.zoom(-0.1)
                elif event.key.keysym.sym == sdl2.SDLK_w:
                    camera.pitch(0.1)
                elif event.key.keysym.sym == sdl2.SDLK_s:
                    camera.pitch(-0.1)
                elif event.key.keysym.sym == sdl2.SDLK_a:
                    camera.yaw(-0.1)
                elif event.key.keysym.sym == sdl2.SDLK_d:
                    camera.yaw(0.1)
                elif event.key.keysym.sym == sdl2.SDLK_q:
                    camera.roll(-0.1)
                elif event.key.keysym.sym == sdl2.SDLK_e:
                    camera.roll(0.1)
                elif event.key.keysym.sym == sdl2.SDLK_PERIOD:
                    STEPS_PER_FRAME += 1
                elif event.key.keysym.sym == sdl2.SDLK_COMMA:
                    STEPS_PER_FRAME = max(0, STEPS_PER_FRAME-1)
                    
        sdl2.SDL_Delay(10)

        world.process()


if __name__ == "__main__":
    sys.exit(run())
