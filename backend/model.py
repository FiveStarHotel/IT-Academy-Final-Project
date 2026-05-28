import tensorflow as tf

class DenseLayer(tf.Module):
    def __init__(self, in_dim: int, out_dim: int, name: str = None):
        super().__init__(name=name)
        self.W = tf.Variable(
            tf.random.normal(shape=(in_dim, out_dim), stddev=tf.math.sqrt(2 / float(in_dim))), 
            dtype=tf.float32
        )
        self.b = tf.Variable(tf.zeros(shape=(out_dim,)), dtype=tf.float32)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, self.W) + self.b

class MLPClassifier(tf.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple, out_dim: int, name: str = None):
        super().__init__(name=name)
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        self.layers = [DenseLayer(dims[i], dims[i+1], name=f"Dense_{i+1}") for i in range(len(dims) - 1)]

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = tf.nn.relu(x)
        x = self.layers[-1](x)
        return x

    def predict(self, x: tf.Tensor):
        x = self.__call__(x)
        x -= tf.reduce_max(x, axis=1, keepdims=True)
        x = tf.exp(x)
        x /= tf.reduce_sum(x, axis=1, keepdims=True)
        return x
