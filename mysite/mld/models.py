from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinLengthValidator
from django.conf import settings

class ML_Type(models.Model):
    index = models.IntegerField()
    title = models.CharField(max_length=128)
    description = models.CharField(max_length=2000, null = True)
    def __str__(self):
        return self.title

class Parameter_Name(models.Model):
    title = models.CharField(max_length=128)
    def __str__(self):
        return self.title

class Data_Type(models.Model):
    title = models.CharField(max_length=128)
    def __str__(self):
        return self.title

class File_Type(models.Model):
    title = models.CharField(max_length=256, null=True, blank=True)
    def __str__(self):
        return self.title

class Dataset(models.Model):
    # index = models.IntegerField()
    title = models.CharField(max_length=128)
    description = models.CharField(max_length=2000, null = True)
    ml_type = models.ForeignKey(ML_Type, on_delete=models.CASCADE)
    data = models.BinaryField(null=True, blank=True, editable=True)
    # data_type = models.CharField(max_length=256, null=True, blank=True)
    file_type = models.ForeignKey(File_Type, on_delete=models.CASCADE, default=0)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    public = models.BooleanField(default=False)
    dimensionality = models.IntegerField(null=True, blank=True)
    number_of_classes = models.IntegerField(null=True, blank=True)
    class_1_count = models.IntegerField(null=True, blank=True)
    class_2_count = models.IntegerField(null=True, blank=True)
    class_1_label = models.CharField(max_length=128, null=True, blank=True)
    class_2_label = models.CharField(max_length=128, null=True, blank=True)
    samples = models.IntegerField(null=True, blank=True)
    target_mean = models.FloatField(null=True, blank=True)
    target_std = models.FloatField(null=True, blank=True)
    target_min = models.FloatField(null=True, blank=True)
    target_max = models.FloatField(null=True, blank=True)
    has_header = models.BooleanField(default=False)
    target_datatype = models.ForeignKey(Data_Type, on_delete=models.CASCADE, default=1)
    # target_datatype = models.CharField(max_length=128, null=True, blank=True, default='int')
    # target_data = models.BinaryField(null=True, blank=True, editable=True)
    def __str__(self):
        return self.title

class Algorithm(models.Model):
    index = models.IntegerField()
    title = models.CharField(max_length=128)
    ml_type = models.ForeignKey(ML_Type, on_delete=models.CASCADE, default=1)
    description = models.CharField(max_length=2000, null = True)
    number_of_parameters = models.IntegerField()
    parameter_1_name = models.ForeignKey(Parameter_Name, on_delete=models.CASCADE, default=1, related_name='algorithm_parameter_1_name')
    parameter_2_name = models.ForeignKey(Parameter_Name, on_delete=models.CASCADE, default=2, related_name='algorithm_parameter_2_name')
    parameter_3_name = models.ForeignKey(Parameter_Name, on_delete=models.CASCADE, default=3, related_name='algorithm_parameter_3_name')
    parameter_1_type = models.ForeignKey(Data_Type, on_delete=models.CASCADE, default=1, related_name='algorithm_parameter_1_type')
    parameter_2_type = models.ForeignKey(Data_Type, on_delete=models.CASCADE, default=2, related_name='algorithm_parameter_2_type')
    parameter_3_type = models.ForeignKey(Data_Type, on_delete=models.CASCADE, default=3, related_name='algorithm_parameter_3_type')
    # parameter_1_name = models.CharField(max_length=128)
    # parameter_2_name = models.CharField(max_length=128, null = True, blank = True)
    # parameter_3_name = models.CharField(max_length=128, null = True, blank = True)
    def __str__(self):
        return self.title

class Entry(models.Model):
    # title = models.CharField(max_length=128, null = True, blank = True, default = "-")
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    ml_type = models.ForeignKey(ML_Type, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    algorithm = models.ForeignKey(Algorithm, on_delete=models.CASCADE)
    parameter_1_value = models.FloatField(default=1)
    parameter_2_value = models.FloatField(null = True, blank = True)
    parameter_3_value = models.FloatField(null = True, blank = True)
    train_accuracy = models.FloatField(null = True, blank = True)
    test_accuracy = models.FloatField(null = True, blank = True)
    r2_train_score = models.FloatField(null = True, blank = True)
    r2_test_score = models.FloatField(null = True, blank = True)
    rmse_train = models.FloatField(null = True, blank = True)
    rmse_test = models.FloatField(null = True, blank = True)
    precision = models.FloatField(null = True, blank = True)
    recall = models.FloatField(null = True, blank = True)
    f1 = models.FloatField(null = True, blank = True)
    time = models.FloatField(null = True, blank = True)
    roc_auc = models.FloatField(null = True, blank = True)
    avg_precision = models.FloatField(null = True, blank = True)
    fpr = models.BinaryField(null=True, blank=True, editable=True)
    tpr = models.BinaryField(null=True, blank=True, editable=True)
    precision_c = models.BinaryField(null=True, blank=True, editable=True)
    recall_c = models.BinaryField(null=True, blank=True, editable=True)
    thresholds = models.BinaryField(null=True, blank=True, editable=True)
    thresholds2 = models.BinaryField(null=True, blank=True, editable=True)
    TN = models.IntegerField(null=True, blank=True)
    TP = models.IntegerField(null=True, blank=True)
    FP = models.IntegerField(null=True, blank=True)
    FN = models.IntegerField(null=True, blank=True)
    target_data = models.BinaryField(null=True, blank=True, editable=True)
    def __str__(self):
        return self.dataset.title + " - " + self.algorithm.title +  ": " + self.algorithm.parameter_1_name.title + " = " + str(self.parameter_1_value)

class Graph(models.Model):
    # title = models.CharField(max_length=128, null = True, blank = True, default = "-")
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    ml_type = models.ForeignKey(ML_Type, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    algorithm = models.ForeignKey(Algorithm, on_delete=models.CASCADE)
    parameter_to_plot = models.IntegerField(default=1)
    def __str__(self):
        return self.dataset.title + " - " + self.algorithm.title