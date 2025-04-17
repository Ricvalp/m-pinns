rm -r ./fit/checkpoints/sphere

PYTHONPATH=. \
python fit/fit_autoencoder.py \
--config=fit/config/fit_autoencoder_sphere.py \
--config.train.reg_lambda=0.1 \
--config.train.lambda_geo_loss=1.