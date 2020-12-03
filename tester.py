from gen_circle import synthetic_gen, oval_gen
from main import model1, plot_pred

# generate new image
x, _ = next(synthetic_gen())
# predict
prediction = model1.predict(x)

# check with oval instead
x1, _ = oval_gen()
prediction1 = model1.predict(x1)

# examine 1 image
im = x1[0]
p = prediction1[0]
plot_pred(im, p)
