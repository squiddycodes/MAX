from django.db import models

# Create your models here.
class Case(models.Model):
  ticker = models.CharField()
  t = models.IntegerField()
  iso_utc = models.DateTimeField() #will look like string from json
  iso_ny = models.DateTimeField() #will look like string from json - X on graph
  o = models.FloatField()
  h = models.FloatField()
  l = models.FloatField()
  c = models.FloatField() # C
  v = models.IntegerField()
  vw = models.FloatField()
  n = models.IntegerField()
  
  span = models.CharField()
  vola = models.FloatField(null=True)
  s = models.FloatField(null=True)
