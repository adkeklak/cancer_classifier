from django.core.management.base import BaseCommand
from cancer_classificator.ml.train import model_test as test_model
        
class Command(BaseCommand):
    help = 'Train the ML model'

    def handle(self, *args, **options):
        test_model()
        
        self.stdout.write(self.style.SUCCESS('Successfully tested the model'))