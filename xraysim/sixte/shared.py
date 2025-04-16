import os

# SIXTE instruments configuration file
instrumentsConfigFile = os.path.join(os.path.dirname(__file__), '../../sixte_instruments.json')

# SIXTE instruments parent directory
instrumentsDir = os.environ.get('SIXTE') + '/share/sixte/instruments'

# Default SIXTE command
defaultSixteCommand = 'sixtesim'

# List of special instruments implemented in xraysim
specialInstrumentsList = ['erosita']
