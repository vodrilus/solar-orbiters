"""My Little Quaternion

A tiny, super-humble and inefficient module to handle quaternions. Inspired by
this lovely article: http://www.3dgep.com/understanding-quaternions/

For implementations closer to production code, visit the Cheese Shop, e.g.
https://pypi.python.org/pypi/Quaternion/ or
https://pypi.python.org/pypi/quaternions/

TODO: Add tests to "main" section.
TODO: Expand methods to accept also non-Quaternions (such as scalars).
TODO: Handle error creep.
TODO: Refactor to override operators.
TODO: Add vector rotation and quaternion interpolation?"""

import math

class Quaternion:
    """A quaternion."""
    def __init__(self, s=0, x=0, y=0, z=0):
        self.s = s
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return 'Quaternion: {} + {}i + {}j + {}k'.format(self.s, self.x,
                                                         self.y, self.z)

    def add(self, other):
        """Return sum of this and another quaternion."""
        if not isinstance(other, Quaternion):
            raise TypeError("other not Quaternion")
            
        s = self.s + other.s
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z

        return Quaternion(s,x,y,z)

    def sub(self, other):
        """Return difference of this and another quaternion."""
        if not isinstance(other, Quaternion):
            raise TypeError("other not Quaternion")
            
        s = self.s - other.s
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z

        return Quaternion(s,x,y,z)

    def mul(self, other):
        """Return product of the quaternion or a scalar or a quaternion."""
        if isinstance(other, (int, float)):
            s = self.s * other
            x = self.x * other
            y = self.y * other
            z = self.z * other
            
            return Quaternion(s, x, y, z)
        
        elif isinstance(other, Quaternion):
            s_1, x_1, y_1, z_1 = self.s, self.x, self.y, self.z
            s_2, x_2, y_2, z_2 = other.s, other.x, other.y, other.z

            s = s_1 * s_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
            x = s_1 * x_2 + x_1 * s_2 + y_1 * z_2 - z_1 * y_2
            y = s_1 * y_2 - x_1 * z_2 + y_1 * s_2 + z_1 * x_2
            z = s_1 * z_2 + x_1 * y_2 - y_1 * x_2 + z_1 * s_2

            return Quaternion(s, x, y, z)
        
        else:
            raise TypeError("Type of other not acceptable.")

    def dot(self, other):
        """Return dot product of both quaternions."""
        if isinstance(other, Quaternion):
            return (self.s * other.s +
                    self.x * other.x +
                    self.y * other.y +
                    self.z * other.z)
        else:
            raise TypeError("other not Quaternion") # TODO: Expand?

    def truediv(self, other):
        """Return true division."""
        if isinstance(other, (int, float)):
            s = self.s / other
            x = self.x / other
            y = self.y / other
            z = self.z / other

            return Quaternion(s, x, y, z)
        elif isinstance(other, Quaternion):
            raise TypeError("Division by quaternion not supported.")
        else:
            raise TypeError("Type of other not acceptable.")

    def abs(self):
        """Return magnitude or norm of the quaternion."""

        return math.sqrt(self.s * self.s +
                         self.x * self.x +
                         self.y * self.y +
                         self.z * self.z)

    def conjugate(self):
        """Return the conjugate of the quaternion (i.e. with negated imaginary
        part)."""
        s = self.s
        x = - self.x
        y = - self.y
        z = - self.z
        
        return Quaternion(s, x, y, z)

    def inverse(self):
        """Return the inverse of the quaternion. I.e. q**(-1)."""
        
        return self.conjugate().truediv(self.abs()*self.abs())

    def real(self):
        """Return the real (or scalar) part of the quaternion."""
        return Quaternion(self.s, 0, 0, 0)
    
    def imaginary(self):
        """Return the imaginary (or vector) part of the quaternion."""
        return Quaternion(0, self.x, self.y, self.z)

    def unit_quaternion(self):
        return self.imaginary().truediv(self.imaginary().abs())

    def norm(self):
        """An alias for Quaternion.abs()"""
        return self.abs()
        
    

if __name__ == '__main__':
    # TODO: Add tests...
    print('My Little Quaternion\n' +
          'A tiny, super-humble and inefficient module to handle quaternions.')
