import csv

input_file = "trades_log.csv"
output_file = "trades_log_fixed.csv"

with open(input_file, "r", encoding="utf-8") as infile, open(
    output_file, "w", newline="", encoding="utf-8"
) as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Filtra filas vacías
        if not row:
            continue

        # Toma solo las 4 primeras columnas (symbol, event, price, timestamp)
        clean_row = row[:4]
        writer.writerow(clean_row)

print("✅ Archivo limpiado guardado como:", output_file)
