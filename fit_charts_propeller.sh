NUM_CHART=38

rm -r fit/checkpoints/propeller/chart_$NUM_CHART

PYTHONPATH=. \
python fit/fit_autodecoder.py \
--config=fit/config/fit_autodecoder_propeller.py \
--config.charts_to_fit='('"$NUM_CHART"')' \
--config.train.reg_lambda=3. \
--config.train.warmup_steps=20000 \
--config.train.lambda_geo_loss=100.