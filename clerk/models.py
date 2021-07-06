from django.db import models


class client(models.Model):
    clientName = models.CharField(max_length=50 , blank=False , null=False)
    img1 = models.ImageField(upload_to='imgs' , blank=True , null=True)
    img2 = models.ImageField(upload_to='imgs' , blank=True , null=True)
    img3 = models.ImageField(upload_to='imgs' , blank=True , null=True)
    verifiedImg = models.ImageField(upload_to='imgs' , blank=True , null=True)
    Bimg1 = models.TextField( blank=True , null=True)
    Bimg2 = models.TextField( blank=True, null=True)
    Bimg3 = models.TextField( blank=True, null=True)
    BverifiedImg = models.TextField( blank=True, null=True)

    class Meta:
        verbose_name = 'client'
        verbose_name_plural = 'clients'

    def __str__(self):
        return str(self.clientName)



