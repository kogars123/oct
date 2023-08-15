import unittest
import paddle
from ivy.functional.frontends.paddle.tensor.manipulation import moveaxis

class TestMoveaxis(unittest.TestCase):
    def test_moveaxis(self):
        # Test case 1: move the first axis to the last position
        x = paddle.randn([3, 4, 5])
        y = moveaxis(x, 0, -1)
        self.assertEqual(y.shape, [4, 5, 3])

        # Test case 2: move the last axis to the first position
        x = paddle.randn([3, 4, 5])
        y = moveaxis(x, -1, 0)
        self.assertEqual(y.shape, [5, 3, 4])

        # Test case 3: move multiple axes
        x = paddle.randn([3, 4, 5, 6])
        y = moveaxis(x, [0, 1], [-1, -2])
        self.assertEqual(y.shape, [5, 6, 4, 3])

if __name__ == '__main__':
    unittest.main()

