# Generated by Django 3.1.4 on 2021-01-02 00:54

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Algorithm',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('index', models.IntegerField()),
                ('title', models.CharField(max_length=128)),
                ('description', models.CharField(max_length=2000, null=True)),
                ('number_of_parameters', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Data_Type',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=128)),
            ],
        ),
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=128)),
                ('description', models.CharField(max_length=2000, null=True)),
                ('data', models.BinaryField(blank=True, editable=True, null=True)),
                ('public', models.BooleanField(default=False)),
                ('dimensionality', models.IntegerField(blank=True, null=True)),
                ('number_of_classes', models.IntegerField(blank=True, null=True)),
                ('class_1_count', models.IntegerField(blank=True, null=True)),
                ('class_2_count', models.IntegerField(blank=True, null=True)),
                ('class_1_label', models.CharField(blank=True, max_length=128, null=True)),
                ('class_2_label', models.CharField(blank=True, max_length=128, null=True)),
                ('samples', models.IntegerField(blank=True, null=True)),
                ('target_mean', models.FloatField(blank=True, null=True)),
                ('target_std', models.FloatField(blank=True, null=True)),
                ('target_min', models.FloatField(blank=True, null=True)),
                ('target_max', models.FloatField(blank=True, null=True)),
                ('has_header', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='File_Type',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(blank=True, max_length=256, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='ML_Type',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('index', models.IntegerField()),
                ('title', models.CharField(max_length=128)),
                ('description', models.CharField(max_length=2000, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Parameter_Name',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=128)),
            ],
        ),
        migrations.CreateModel(
            name='Graph',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('parameter_to_plot', models.IntegerField(default=1)),
                ('algorithm', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mld.algorithm')),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mld.dataset')),
                ('ml_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mld.ml_type')),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Entry',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('parameter_1_value', models.FloatField(default=1)),
                ('parameter_2_value', models.FloatField(blank=True, null=True)),
                ('parameter_3_value', models.FloatField(blank=True, null=True)),
                ('train_accuracy', models.FloatField(blank=True, null=True)),
                ('test_accuracy', models.FloatField(blank=True, null=True)),
                ('r2_train_score', models.FloatField(blank=True, null=True)),
                ('r2_test_score', models.FloatField(blank=True, null=True)),
                ('rmse_train', models.FloatField(blank=True, null=True)),
                ('rmse_test', models.FloatField(blank=True, null=True)),
                ('precision', models.FloatField(blank=True, null=True)),
                ('recall', models.FloatField(blank=True, null=True)),
                ('f1', models.FloatField(blank=True, null=True)),
                ('time', models.FloatField(blank=True, null=True)),
                ('roc_auc', models.FloatField(blank=True, null=True)),
                ('avg_precision', models.FloatField(blank=True, null=True)),
                ('fpr', models.BinaryField(blank=True, editable=True, null=True)),
                ('tpr', models.BinaryField(blank=True, editable=True, null=True)),
                ('precision_c', models.BinaryField(blank=True, editable=True, null=True)),
                ('recall_c', models.BinaryField(blank=True, editable=True, null=True)),
                ('thresholds', models.BinaryField(blank=True, editable=True, null=True)),
                ('thresholds2', models.BinaryField(blank=True, editable=True, null=True)),
                ('TN', models.IntegerField(blank=True, null=True)),
                ('TP', models.IntegerField(blank=True, null=True)),
                ('FP', models.IntegerField(blank=True, null=True)),
                ('FN', models.IntegerField(blank=True, null=True)),
                ('target_data', models.BinaryField(blank=True, editable=True, null=True)),
                ('algorithm', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mld.algorithm')),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mld.dataset')),
                ('ml_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mld.ml_type')),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.AddField(
            model_name='dataset',
            name='file_type',
            field=models.ForeignKey(default=0, on_delete=django.db.models.deletion.CASCADE, to='mld.file_type'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='ml_type',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mld.ml_type'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='owner',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='dataset',
            name='target_datatype',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='mld.data_type'),
        ),
        migrations.AddField(
            model_name='algorithm',
            name='ml_type',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='mld.ml_type'),
        ),
        migrations.AddField(
            model_name='algorithm',
            name='parameter_1_name',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='algorithm_parameter_1_name', to='mld.parameter_name'),
        ),
        migrations.AddField(
            model_name='algorithm',
            name='parameter_1_type',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='algorithm_parameter_1_type', to='mld.data_type'),
        ),
        migrations.AddField(
            model_name='algorithm',
            name='parameter_2_name',
            field=models.ForeignKey(default=2, on_delete=django.db.models.deletion.CASCADE, related_name='algorithm_parameter_2_name', to='mld.parameter_name'),
        ),
        migrations.AddField(
            model_name='algorithm',
            name='parameter_2_type',
            field=models.ForeignKey(default=2, on_delete=django.db.models.deletion.CASCADE, related_name='algorithm_parameter_2_type', to='mld.data_type'),
        ),
        migrations.AddField(
            model_name='algorithm',
            name='parameter_3_name',
            field=models.ForeignKey(default=3, on_delete=django.db.models.deletion.CASCADE, related_name='algorithm_parameter_3_name', to='mld.parameter_name'),
        ),
        migrations.AddField(
            model_name='algorithm',
            name='parameter_3_type',
            field=models.ForeignKey(default=3, on_delete=django.db.models.deletion.CASCADE, related_name='algorithm_parameter_3_type', to='mld.data_type'),
        ),
    ]