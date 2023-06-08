import numpy as np

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)

class BackPropNeuralNetwork:
    # resolution - разрешение изображения
    # k - количество нейронов на выходе сети
    # s - количество нейронов скрытого слоя
    # activationFunc - функция активации
    def __init__(self, resolution, k, s, activationFunc):
        self.resolution = resolution
        self.k = k
        self.s = s
        self.activationFunc = activationFunc
        self.__init_weights()

    # Инициализация весов
    def __init_weights(self):
        self.hidden_layer_weights = np.random.rand(self.resolution, self.s) * 0.1
        self.output_layer_weights = np.random.rand(self.s, self.k) * 0.1
        self.bias = np.random.randn(self.s) * 0.1

    def __calc_hide_layer_output(self, x):
        hide_layer_output = np.matmul(x, self.hidden_layer_weights) + self.bias
        hide_layer_output[hide_layer_output < 0] = 0
        return hide_layer_output

    def __calc_hide_layer_derivative(self, hide_layer_output):
        hide_layer_output[hide_layer_output > 0] = 1
        return hide_layer_output

    # Прямой проход:
    #   1. Вычисление значений выходных сигналов нейронов всех слоев
    #   2. Вычисление значений производных функций активации на каждом слое сети
    def __forward(self, x):
        # выход скрытого слоя
        hide_layer_output = self.__calc_hide_layer_output(x)
        # производная выхода скрытого слоя
        derivative = self.__calc_hide_layer_derivative(hide_layer_output)
        # выход сети
        output = self.activationFunc(np.matmul(hide_layer_output, self.output_layer_weights))
        return output, hide_layer_output, derivative

    # Обратный проход:
    #   1. Вычисление значения целевой функции и ее градиента
    #   2. Корректировка синаптических весов
    def __reverse(self, trainX, trainY, currentOutput, current_hide_layer_output, derivative, alpha):
        error = currentOutput - trainY
        grad_output = np.matmul(current_hide_layer_output.T, error)

        error_hide_layer = np.matmul(error, self.output_layer_weights.T) * derivative
        grad_hide_layer = np.matmul(trainX.T, error_hide_layer)

        delta_output = grad_output / self.resolution + alpha * self.output_layer_weights
        delta_hidden = grad_hide_layer / self.resolution + alpha * self.hidden_layer_weights
        delta_bias = np.mean(error_hide_layer, axis=0)

        return delta_output, delta_hidden, delta_bias

    # Обработать входные данные
    def handleInput(self, x):
        result, _, _ = self.__forward(x)
        return result

    def train(self, trainX, trainY, stepCount=1500, acceptableError=0.005):
        self.__init_weights()
        error_history = []
        for i in range(stepCount):
            output, hide_layer_output, derivative = self.__forward(trainX)
            error = np.mean(np.abs(output - trainY))
            print(f'iteration: {i}/{stepCount}, current error: {error}')
            error_history.append(error)
            if error <= acceptableError:
                break
            delta_output, delta_hidden, delta_bias = self.__reverse(trainX, trainY, output, hide_layer_output,
                                                                    derivative, alpha)
            # lr = 0.1
            # self.hidden_layer_weights -= lr*delta_hidden
            # self.output_layer_weights -= lr*delta_output
            # self.bias -= lr*delta_bias

            # RMSProp
            gamma = 0.99
            eps = 1e-3
            eta = 5e-1
            ghidden = goutput = gb = 1
            ghidden = gamma * ghidden + (1 - gamma) * np.sum(delta_hidden ** 2)
            eta_hidden = eta / np.sqrt(ghidden + eps)
            self.hidden_layer_weights -= eta_hidden * delta_hidden
            goutput = gamma * goutput + (1 - gamma) * np.sum(delta_output ** 2)
            eta_output = eta / np.sqrt(goutput + eps)
            self.output_layer_weights -= eta_output * delta_output
            gb = gamma * gb + (1 - gamma) * np.sum(delta_bias ** 2)
            etab = eta / np.sqrt(gb + eps)
            self.bias -= etab * delta_bias

            randomize = np.arange(trainX.shape[0])
            np.random.shuffle(randomize)
            trainX = trainX[randomize]
            trainY = trainY[randomize]
        return error_history


