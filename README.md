# TextGan

Make it great again!
### Get started

Download the labels dataset (Repo is private so no worries with regards to privacy)

```sh
$ git clone https://github.com/PhilipBotros/TextGan
$ chmod +x get_data.sh
$ ./get_data.sh
```

### Todo's
  - Build an oracle model LSTM (from SeqGan) for pure error comparison
  - Finish sampling operation for attention generator
  - Compare attention generator with basic generator on time-per-epoch
  - Compare attention generator with basic generator on log-likelihood
  - Compare attention generator with basic generator on sample output
  - Integrate pretrain and train scripts into a single master script
  - Enable settings of different types of generators
  - Get pytorch to run on Das-4
  - Vary attention lookback time
