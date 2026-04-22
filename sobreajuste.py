import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

JSON_PATH = "historico_experimentos.json"
WINDOW = 5
ALPHA = 0.05

# ============================================================
# CARGA DEL ÚLTIMO EXPERIMENTO
# ============================================================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    exp = json.load(f)[-1]

batch_eval = exp["batch_eval"]

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def es_constante(x):
    x = np.asarray(x, dtype=float)
    return np.allclose(x, x[0])

def comprobar_normalidad(x):
    x = np.asarray(x, dtype=float)

    if es_constante(x):
        return False, np.nan   # no evaluable con Shapiro

    stat, p = shapiro(x)
    return p > ALPHA, p

def elegir_prueba(x, y):
    """
    Devuelve:
    - nombre de la prueba
    - p-valor
    - p de Shapiro para x
    - p de Shapiro para y
    - p de Levene
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    normal_x, p_shapiro_x = comprobar_normalidad(x)
    normal_y, p_shapiro_y = comprobar_normalidad(y)

    p_levene = np.nan

    # Si ambos son normales, usar t
    if normal_x and normal_y:
        _, p_levene = levene(x, y)

        if p_levene > ALPHA:
            # t de Student
            _, p = ttest_ind(x, y, equal_var=True)
            prueba = "t de Student"
        else:
            # t de Welch
            _, p = ttest_ind(x, y, equal_var=False)
            prueba = "t de Welch"
    else:
        # no normalidad -> no paramétrica
        _, p = mannwhitneyu(x, y, alternative="two-sided")
        prueba = "Mann-Whitney U"

    return prueba, p, p_shapiro_x, p_shapiro_y, p_levene

# ============================================================
# ANÁLISIS CADA 5 ÉPOCAS
# ============================================================
resultados = []

for i in range(len(batch_eval) - WINDOW + 1):
    bloque = batch_eval[i:i + WINDOW]

    epoch_inicio = bloque[0]["epoch"]
    epoch_fin = bloque[-1]["epoch"]

    train_acc = np.concatenate([e["train_acc_batches"] for e in bloque])
    dev_acc   = np.concatenate([e["dev_acc_batches"] for e in bloque])

    train_loss = np.concatenate([e["train_loss_batches"] for e in bloque])
    dev_loss   = np.concatenate([e["dev_loss_batches"] for e in bloque])

    prueba_acc, p_acc, p_sh_train_acc, p_sh_dev_acc, p_lev_acc = elegir_prueba(train_acc, dev_acc)
    prueba_loss, p_loss, p_sh_train_loss, p_sh_dev_loss, p_lev_loss = elegir_prueba(train_loss, dev_loss)

    media_train_acc = np.mean(train_acc)
    media_dev_acc = np.mean(dev_acc)
    media_train_loss = np.mean(train_loss)
    media_dev_loss = np.mean(dev_loss)

    significativo = (
        media_train_acc > media_dev_acc and
        media_dev_loss > media_train_loss and
        p_acc < ALPHA and
        p_loss < ALPHA
    )

    resultados.append({
        "inicio": epoch_inicio,
        "fin": epoch_fin,
        "train_acc": media_train_acc,
        "dev_acc": media_dev_acc,
        "train_loss": media_train_loss,
        "dev_loss": media_dev_loss,
        "prueba_acc": prueba_acc,
        "p_acc": p_acc,
        "p_sh_train_acc": p_sh_train_acc,
        "p_sh_dev_acc": p_sh_dev_acc,
        "p_lev_acc": p_lev_acc,
        "prueba_loss": prueba_loss,
        "p_loss": p_loss,
        "p_sh_train_loss": p_sh_train_loss,
        "p_sh_dev_loss": p_sh_dev_loss,
        "p_lev_loss": p_lev_loss,
        "significativo": significativo
    })

# ============================================================
# MOSTRAR RESULTADOS
# ============================================================
print("\nANÁLISIS DEL SOBREAJUSTE CADA 5 ÉPOCAS\n")

epoca_sobreajuste = None

for r in resultados:
    print(f"Épocas {r['inicio']}-{r['fin']}:")
    print(f"  Train acc = {r['train_acc']:.4f} | Dev acc = {r['dev_acc']:.4f}")
    print(f"  Shapiro accuracy -> train p = {r['p_sh_train_acc']:.4f} | dev p = {r['p_sh_dev_acc']:.4f}")
    print(f"  Levene accuracy  -> p = {r['p_lev_acc']:.4f}")
    print(f"  Prueba accuracy  -> {r['prueba_acc']} | p = {r['p_acc']:.4f}")

    print(f"  Train loss = {r['train_loss']:.4f} | Dev loss = {r['dev_loss']:.4f}")
    print(f"  Shapiro loss -> train p = {r['p_sh_train_loss']:.4f} | dev p = {r['p_sh_dev_loss']:.4f}")
    print(f"  Levene loss  -> p = {r['p_lev_loss']:.4f}")
    print(f"  Prueba loss  -> {r['prueba_loss']} | p = {r['p_loss']:.4f}")

    print(f"  Diferencia significativa: {r['significativo']}\n")

    if epoca_sobreajuste is None and r["significativo"]:
        epoca_sobreajuste = r["fin"]

if epoca_sobreajuste is not None:
    print(f"Inicio probable del sobreajuste: época {epoca_sobreajuste}")
else:
    print("No se ha detectado sobreajuste significativo con este criterio.")

# ============================================================
# GRÁFICA RESUMEN
# ============================================================
x = [r["fin"] for r in resultados]
train_acc_y = [r["train_acc"] for r in resultados]
dev_acc_y = [r["dev_acc"] for r in resultados]
train_loss_y = [r["train_loss"] for r in resultados]
dev_loss_y = [r["dev_loss"] for r in resultados]

plt.figure(figsize=(10, 6))
plt.plot(x, train_acc_y, "o-", label="Train accuracy (media 5 épocas)")
plt.plot(x, dev_acc_y, "o-", label="Dev accuracy (media 5 épocas)")
plt.plot(x, train_loss_y, "s--", label="Train loss (media 5 épocas)")
plt.plot(x, dev_loss_y, "s--", label="Dev loss (media 5 épocas)")

if epoca_sobreajuste is not None:
    plt.axvline(epoca_sobreajuste, color="red", linestyle=":", linewidth=2,
                label=f"Inicio sobreajuste: época {epoca_sobreajuste}")

plt.xlabel("Época final de cada bloque de 5")
plt.ylabel("Valor")
plt.title("Detección de sobreajuste cada 5 épocas")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("apartado5_sobreajuste.png", dpi=300)
plt.show()
