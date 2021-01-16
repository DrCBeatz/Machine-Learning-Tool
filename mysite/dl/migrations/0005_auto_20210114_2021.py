# Generated by Django 3.1.4 on 2021-01-14 20:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dl', '0004_image_gallery'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='image',
            name='thumbnail',
        ),
        migrations.AlterField(
            model_name='image',
            name='image',
            field=models.ImageField(default='', upload_to='images'),
        ),
    ]