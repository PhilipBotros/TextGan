# TextGan
A natural language generator trained by the adverserial framework.
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
Add attention to the sequence model by:
```sh
$ python main.py --attention
```
We first pre train the generator/discriminator to obtain a vanilla language model after which we start the adverserial training. The rest of the (hyper) parameters can be found in settings.py and can be given as command line arguments.
