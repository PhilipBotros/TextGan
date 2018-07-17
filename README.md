# TextGan
### Get started

Download the labels dataset:

```sh
$ git clone https://github.com/PhilipBotros/TextGan
$ chmod +x get_data.sh
$ ./get_data.sh
```

### Run
Run the main model with either rewards coming from the LSTM directly or through Monte Carlo rollout by varying:
```sh
$ python main.py --lstm_rewards
```
Rest of the (hyper) parameters can be found in settings.py and can be given as command line arguments.
