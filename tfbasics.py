import tensorflow as tf #导入tensorflow包

x1=tf.constant([5])#对常量5进行声明
x2=tf.constant([6])#对常量6进行声明

result1 = tf.multiply(x1,x2)#在tensorflow中的两个常量进行相乘
print(result1)


sess=tf.Session()#创建一个session
print(sess.run(result1))#让tf在session中运行，表示tensor里面的值
sess.close()

#更快速的写法
with tf.Session() as sess:
	print(sess.run(result1))