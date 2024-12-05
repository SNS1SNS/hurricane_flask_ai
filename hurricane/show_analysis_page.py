from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def show_analysis_page(self):
    for widget in self.root.winfo_children():
        widget.destroy()

    # Заголовок
    header = ctk.CTkLabel(self.root, text="Hurricane Analysis", font=("Helvetica", 20, "bold"))
    header.pack(pady=30)

    frame = ctk.CTkFrame(self.root, corner_radius=10)
    frame.pack(pady=20, padx=20)

    ctk.CTkButton(frame, text="Back to Main", command=self.show_main_page, width=150).pack(pady=10)

    # Анализ данных
    df = pd.read_csv("Hurricane_Data.csv")

    # Распределение категорий
    figure1 = Figure(figsize=(5, 4), dpi=100)
    ax1 = figure1.add_subplot(111)
    df['CycloneCategory'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Cyclone Category Distribution')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Frequency')

    # Связь температуры и скорости ветра
    figure2 = Figure(figsize=(5, 4), dpi=100)
    ax2 = figure2.add_subplot(111)
    ax2.scatter(df['AirTemperature_C'], df['WindSpeed_kmh'], color='red', alpha=0.5)
    ax2.set_title('Air Temperature vs Wind Speed')
    ax2.set_xlabel('Air Temperature (°C)')
    ax2.set_ylabel('Wind Speed (km/h)')

    # Отображение графиков
    canvas1 = FigureCanvasTkAgg(figure1, self.root)
    canvas1.get_tk_widget().pack(pady=10)

    canvas2 = FigureCanvasTkAgg(figure2, self.root)
    canvas2.get_tk_widget().pack(pady=10)
