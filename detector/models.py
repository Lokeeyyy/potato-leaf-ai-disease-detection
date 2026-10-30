
from django.db import models


class PredictionHistory(models.Model):
    disease = models.CharField(max_length=100)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.disease
