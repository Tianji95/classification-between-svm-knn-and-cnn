# classification-between-svm-knn-and-cnn
A contrast between three classification algorithm：svm knn and cnn

In order to know the difference of the three algorithm  svm knn and cnn， I write some code to implement a classification using these three algorithm.
Had it to say, I implement svm and knn using matlab, while implement cnn using tensorflow. so this compare is not fare in some degree.But anyway, we can get some imformation from this contrast.

the dataset I use the affNIST in [!http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/

**SVM**

SVM is a binary classify algorithm, so to make a 10 classes classification, we must implement a one-to-one svm classification. 
This method will slow down the SVM heavily. the result given by the SVM is:

training data: 100 000

test data: 20 000

training error:14.294%

test error: 14.15%

training time:6048 seconds

**KNN**

While KNN itself is a Multi-classification algorithm, so we just adjust the data to the KNN and then we will get the result. 
But the problem is that it is difficute to choose the suitable parameter K. So I run this algorithm several times, here is the result

training data: 100 000

test data: 20 000

K:20

test error: 27.6%

training time:21660 seconds

K:10

test error: 24.4%

training time:19712 seconds

K:5

test error: 22.5%

training time:20478 seconds

K:2

test error: 24.3%

training time:21459 seconds

**CNN**

I use tensorflow to implement CNN, and use GTX1070 to run the tensorflow code. So theoretically CNN will much faster than SVM and KNN.
While actually is over 50 times faster than one-to-one SVM.It used about 10 second to train 100000 data. here is the detail:

training data: 100 000

test data: 20 000

training error:23%

test error: 24%

training time:about 12 seconds


training data: 1 600 000

test data: 640 000

training error:2%

test error: 2.3%

training time:about 350 seconds
