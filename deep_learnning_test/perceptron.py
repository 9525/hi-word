from functools import reduce
#定义一个感知器
class Perceptron(object):
    def __init__(self,input_num,activator):
        #初始化
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        """返回一个对象的描述信息"""
        # print(num)
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self,input_vec):
        #输入向量，输出感知器的结果
        #sum = input_vec[0] * self.weights[0] + input_vec[1] * self.weights[1] + self.bias
        sum = self.bias
        for index in range(len(input_vec)):
            sum+=input_vec[index] * self.weights[index]
        #激活函数
        return  self.activator(sum)


    def train(self,input_vecs,labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self,input_vecs, labels, rate):
        for index in range(len(input_vecs)):
            output = self.predict(input_vecs[index])
            self._updata_weight(input_vecs[index],output,labels[index],rate)

    def _updata_weight(self,input_vec,output,label,rate):
        delta = label - output
        print("delta:", delta)
        for index in range(len(input_vec)):
            self.weights[index] = (rate * (delta * input_vec[index])) + self.weights[index]
        self.bias += rate*delta

def f(x):
    return 1 if x > 0 else 0

def get_training_dataset():
    input_vecs = [(1,1),(0,0),(1,0),(0,1)]
    lables = [1,0,0,0]
    return input_vecs,lables

def train_and_perceptron():
    p = Perceptron(2, f)
    input_vecs, lables = get_training_dataset()
    p.train(input_vecs, lables , 10, 0.1)
    return p

if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception)
# 测试
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))