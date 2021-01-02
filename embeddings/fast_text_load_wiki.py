import fasttext

model = fasttext.load_model("wiki.el.bin")
print(model)

vec = model.get_word_vector('τσιρκο')
nn = model.get_nearest_neighbors('ξεφτιλες')

print(vec)
print(nn)
