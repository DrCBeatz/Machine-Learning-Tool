# Generated by Django 3.1.4 on 2021-01-09 20:42

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dl', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='image',
            name='image_type',
        ),
    ]
