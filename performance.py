import json
import matplotlib.pyplot as plt

with open("historico_experimentos.json", "r", encoding="utf-8") as f:
    exp = json.load(f)[-1]

plt.figure(figsize=(10, 6))

plt.plot(exp["epochs"], exp["accuracy"], "o-", label="Train accuracy")
plt.plot(exp["epochs"], exp["val_accuracy"], "o-", label="Dev accuracy")
plt.plot(exp["epochs"], exp["loss"], "s--", label="Train loss")
plt.plot(exp["epochs"], exp["val_loss"], "s--", label="Dev loss")

plt.xlabel("Época")
plt.ylabel("Valor")
plt.title("Evolución de accuracy y loss en train y dev")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("apartado3_train_dev.png", dpi=300)
plt.show()