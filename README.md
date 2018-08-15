# spare gcca

## Structure of Project 

The structure of this project should be contruct like this:

- your_project_name/
	- gcca/
		- spare_gcca
		- origin_gcca
		- ...
	- gcca_data/
		- csv_data/
		- genes_data/
		- twitter_data/
		- three_view_data/
		- weight/synthetic_data_model_list_U.pickle
		- image/plot_for_spare

## Package

You should install some necessary package within python3

```python
keras
theano
numpy
sklearn
pandas
```

## Test result

Main programe is in `all_kinds_of_test.py`. You should open this file and choose which model you want to run

Choose whatever you want and have fun:
```python
Which model do you want to choose?
0.gcca,
1.spare_gcca,
2.cca,
3.deepcca,
4.WeightedGCCA,
5.dgcca_
>>>0
Which result do you want to test?
0.test in gene data,
1.test in gene data for std,
2.test in gene data whether normalize or not,
4. nothing
>>>0
```

And also, you can check the file which you like and deepen it.

## parameter tunning

In deep cca, you only can tune `epoch` , `batch size` and `learning rate`, because the other parameter do not make a big deal. Or maybe you can edit the code whatever you want:

In `deep_cca.py`:
```python
class deepcca(metric):
    def __init__(self, ds, m_rank, batch_size = 50, epoch_num = 10, learning_rate = 1e-3):
        
		# ...

        # parameter you can tune
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
```

In `dgcca_format.py`, you only can tune `epoch` and `batch size`:

```python
class dgcca_(metric):
    def __init__(self, ds, m_rank, batchSize=40, epochs = 200):
        
        # ...

        # parameter you can tune
        self.batchSize = batchSize
        self.epochs = epochs
```

## Feedback

If you have any issue, please let me know or email me `b.ben.hjy@gmail.com` or `879837607@qq.com`

## Reference

Thanks to the code of [wgcca](https://github.com/abenton/wgcca) written by abenton and [deep cca](https://github.com/VahidooX/DeepCCA) written by VahidooX