# Generated by Django 3.2.12 on 2022-05-13 14:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('nasbench101', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='nasbench101result',
            name='image',
            field=models.ImageField(null=True, upload_to=''),
        ),
    ]
