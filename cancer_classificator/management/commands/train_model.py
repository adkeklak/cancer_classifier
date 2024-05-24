from django.core.management.base import BaseCommand
from cancer_classificator.ml.train import train as train_model
from cancer_classificator.ml.model import LungsModel, MlModel

class Command(BaseCommand):
    help = 'Train the ML model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model_type',
            type=str,
            default='lungs',
            help='Type of the model to train (default: lungs)'
        )
        parser.add_argument(
            '--model_name',
            type=str,
            default='lungs',
            help='Name of the model (default: lungs)'
        )
        parser.add_argument(
            '--path',
            type=str,
            default='cancer_classificator/ml/models',
            help='Path to save the model (default: cancer_classificator/ml/models)'
        )
        parser.add_argument(
            '--model_version',
            type=str,
            default='0.1',
            help='Version of the model (default: 0.1)'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=10,
            help='Number of epochs to train the model (default: 10)'
        )
        parser.add_argument(
            '--verbose',
            type=int,
            choices=[0, 1, 2],
            help='Verbosity level: 0=minimal output, 1=normal output, 2=verbose output',
            default=1
        )
        parser.add_argument(
            '--from_file',
            action='store_true',
            help='Load data from file, overwrites other options (default: False)'
        )

    def handle(self, *args, **options):

        model_type_str = options['model_type']
        if model_type_str == 'lungs':
            model_type = LungsModel()
        else:
            model_type = MlModel()

        model_name = options['model_name']
        path = options['path']
        version = options['model_version']
        epochs = options['epochs']
        verbose = options['verbose']
        from_file = options['from_file']

        train_model(
            model_type=model_type,
            model_name=model_name,
            path=path,
            version=version,
            epochs=epochs,
            verbose=verbose,
            from_file=from_file
        )
        
        self.stdout.write(self.style.SUCCESS('Successfully trained the model'))