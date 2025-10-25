from django.db import models

# Create your models here.


class Stock(models.Model):
  ticker = models.CharField()

class Case(models.Model):
  ticker = models.CharField()
  t = models.IntegerField()
  iso_utc = models.DateTimeField() #will look like string from json
  iso_ny = models.DateTimeField() #will look like string from json - X on graph
  o = models.FloatField()
  h = models.FloatField()
  l = models.FloatField()
  c = models.FloatField() # Y 
  v = models.IntegerField()
  vw = models.FloatField()
  n = models.IntegerField()
  
  span = models.CharField()
  vola = models.FloatField(null=True)
  s = models.FloatField(null=True)

  stock = models.ForeignKey(Stock, on_delete=models.CASCADE, null=True)#stock assigned by ticker charfield, matches with name
