"""My Little Quaternion

A tiny, super-humble and inefficient module to handle quaternions for Python3.
Inspired by this lovely article: http://www.3dgep.com/understanding-quaternions/

For implementations closer to production code, visit the Cheese Shop, e.g.
https://pypi.python.org/pypi/Quaternion/ or
https://pypi.python.org/pypi/quaternions/

TODO: Add tests to "main" section.
TODO: Expand methods to accept also non-Quaternions (such as scalars).
TODO: Handle error creep.
TODO: Add quaternion interpolation?
TODO: Improve error messages.
TODO: rotate: Expand vector and axis types to Vector3 and tuple."""

import math
from collections import namedtuple

class Quaternion(namedtuple('Quaternion', 's x y z')):
    """An immutable quaternion class derived from namedtuple.

    Constructor arguments should be integers or floats. This is not enforced,
    so using other types may lead to weird behaviour."""

    def __str__(self):
        return 'Quaternion: [{}, {}i, {}j, {}k]'.format(self.s, self.x,
                                                         self.y, self.z)

    def __add__(self, other):
        """Return the sum of this and another quaternion."""
        if not isinstance(other, Quaternion):
            raise TypeError("Other is not a Quaternion.")
            
        s = self.s + other.s
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z

        return Quaternion(s,x,y,z)

    def __sub__(self, other):
        """Return the difference of this and another quaternion."""
        if not isinstance(other, Quaternion):
            raise TypeError("Other is not a Quaternion.")
            
        s = self.s - other.s
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z

        return Quaternion(s,x,y,z)

    def __matmul__(self, other):
        """Return the (matrix) product of this and another quaternion."""
        if isinstance(other, Quaternion):
            s_1, x_1, y_1, z_1 = self.s, self.x, self.y, self.z
            s_2, x_2, y_2, z_2 = other.s, other.x, other.y, other.z

            s = s_1 * s_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
            x = s_1 * x_2 + x_1 * s_2 + y_1 * z_2 - z_1 * y_2
            y = s_1 * y_2 - x_1 * z_2 + y_1 * s_2 + z_1 * x_2
            z = s_1 * z_2 + x_1 * y_2 - y_1 * x_2 + z_1 * s_2

            return Quaternion(s, x, y, z)
        
        else:
            raise TypeError("Type of other not acceptable.")

    def __mul__(self, other):
        """Return elementwise (dot) product of this and either another
        quaternion or a scalar (integer or float)."""
        if isinstance(other, Quaternion):
            return (self.s * other.s +
                    self.x * other.x +
                    self.y * other.y +
                    self.z * other.z)
        elif isinstance(other, (int, float)):
            s = self.s * other
            x = self.x * other
            y = self.y * other
            z = self.z * other
            
            return Quaternion(s, x, y, z)
        else:
            raise TypeError("Type of other not acceptable.")

    def __rmul__(self, other):
        """Return reversed product (is commutative) of quaternion and scalar."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Return true division."""
        if isinstance(other, (int, float)):
            s = self.s / other
            x = self.x / other
            y = self.y / other
            z = self.z / other

            return Quaternion(s, x, y, z)
        elif isinstance(other, Quaternion):
            # Quaternion division is not mathematically defined
            raise TypeError("Division by quaternion not supported.")
        else:
            raise TypeError("Type of other not acceptable.")

    def __floordiv__(self, other):
        """Return floor division. Might not make sense."""
        if isinstance(other, (int, float)):
            s = self.s // other
            x = self.x // other
            y = self.y // other
            z = self.z // other

            return Quaternion(s, x, y, z)
        elif isinstance(other, Quaternion):
            # Quaternion division is not mathematically defined
            raise TypeError("Division by quaternion not supported.")
        else:
            raise TypeError("Type of other not acceptable.")

    def __abs__(self):
        """Return magnitude or norm of the quaternion."""

        return math.sqrt(self.__mul__(self))

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
        # Avoid redundant sqrt and ** 2
        return self.conjugate().__truediv__(self.__mul__(self))

    def real(self):
        """Return the real (or scalar) part of the quaternion."""
        return Quaternion(self.s, 0, 0, 0)
    
    def imaginary(self):
        """Return the imaginary (or 3d vector) part of the quaternion."""
        return Quaternion(0, self.x, self.y, self.z)

    def normalize(self):
        return self.__truediv__(self.__abs__())

    def norm(self):
        """An alias for Quaternion.abs()"""
        return self.__abs__()

    def __lt__(self, other):
        """Return true if the magnitude of the quaternion is less than the
        magnitude of the compared quaternion or scalar."""
        if not isinstance(other, (int, float, Vector3)):
            raise TypeError("Type of other not acceptable.")
            
        return self.__mul__(self) < other.__mul__(other)

    def __le__(self, other):
        """Return true if the magnitude of the quaternion is less than or equal
        to the magnitude of the compared quaternion or scalar."""
        if not isinstance(other, (int, float, Vector3)):
            raise TypeError("Type of other not acceptable.")
            
        return self.__mul__(self) <= other.__mul__(other)

    def __eq__(self, other):
        """Return true if the magnitude of the quaternion is equal to the
        magnitude of the compared quaternion or scalar."""
        if not isinstance(other, (int, float, Vector3)):
            raise TypeError("Type of other not acceptable.")
            
        return self.__mul__(self) == other.__mul__(other)

    def __ne__(self, other):
        """Return true if the magnitude of the quaternion is not equal to the
        magnitude of the compared quaternion or scalar."""
        if not isinstance(other, (int, float, Vector3)):
            raise TypeError("Type of other not acceptable.")
            
        return self.__mul__(self) != other.__mul__(other)

    def __ge__(self, other):
        """Return true if the magnitude of the quaternion is greater than or
        equal to the magnitude of the compared quaternion or scalar."""
        if not isinstance(other, (int, float, Vector3)):
            raise TypeError("Type of other not acceptable.")
            
        return self.__mul__(self) >= other.__mul__(other)

    def __gt__(self, other):
        """Return true if the magnitude of the quaternion is greater than the
        magnitude of the compared quaternion or scalar."""
        if not isinstance(other, (int, float, Vector3)):
            raise TypeError("Type of other not acceptable.")
            
        return self.__mul__(self) > other.__mul__(other)


def rotate(vector, axis, angle):
    """Rotate given 3d vector (given as an imaginary quaternion) along an axis
    (given as an imaginary quaternion) by angle. Beware of rounding errors.

    TODO: Expand vector and axis types to Vector3 and tuple."""
    half_angle_cos = math.cos(angle/2)
    half_angle_sin = math.sin(angle/2)
    q = Quaternion(half_angle_cos,0,0,0) + half_angle_sin * axis.normalize()
    q_inverse = q.conjugate() # Since q is a unit quaternion, q**(-1) == g*
    return q @ vector @ q_inverse

def angle(vector1, vector2):
    """Return the angle between vector1 and vector2 in radians [0, pi]."""
    u1 = vector1.normalize()
    u2 = vector2.normalize()
    cosine = u1 * u2
    if cosine == 1.0:
        return 0
    elif cosine == -1.0:
        return math.pi
    elif cosine == 0.0:
        return math.pi/2

    # Unit vectors: sine == sqrt(1.0 ** 2 - cosine ** 2) if angle within [0, pi]
    tangent = math.sqrt(1.0 - cosine * cosine) / cosine
    
    return math.atan(tangent)


if __name__ == '__main__':
    # TODO: Add tests...
    print('My Little Quaternion\n' +
          'A tiny, super-humble and inefficient module to handle quaternions.')
