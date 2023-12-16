from nicegui import ui
import numpy as np
import matplotlib.pyplot as plt

ui.label("Manipulator Planner")

a = 5
print(f"==>> a: {a}")

def change_val(v):
    v = v+1
    print(f"==>> v: {v}")

with ui.row():
    with ui.card().classes():
        ui.image('./test/img.jpg').classes('w-full h-20')
        ui.label('this is image')
        ui.button('Click me', on_click=lambda: ui.notify("Clicked", position='bottom-right'))

    with ui.card():
        ui.image('./test/img.jpg')
        ui.label('this is image')

    with ui.card():
        ui.number(label='Number', value=3.1415927, format='%.2f', on_change=lambda e: result.set_text(f'you entered: {e.value}'))
        result = ui.label()

        toggle1 = ui.toggle([1, 2, 3], value=1)
        radio1 = ui.radio([1, 2, 3], value=1).props('inline')
        select1 = ui.select([1, 2, 3], value=1)
        checkbox = ui.checkbox('check me', on_change=lambda: change_val(a))
        switch = ui.switch('switch me')
        slider = ui.slider(min=0, max=100, value=50)
        ui.label().bind_text_from(slider, 'value')

    with ui.card():
        with ui.pyplot(figsize=(3, 2)):
            x = np.linspace(0.0, 5.0)
            y = np.cos(2 * np.pi * x) * np.exp(-x)
            plt.plot(x, y, '-')

    with ui.card():
        columns = [
            {
                'name': 'name',
                'label': 'Name',
                'field': 'name',
                'required': True,
                'align': 'left'
            },
            {
                'name': 'age',
                'label': 'Age',
                'field': 'age',
                'sortable': True
            },
        ]
        rows = [
            {
                'name': 'Alice',
                'age': 18
            },
            {
                'name': 'Bob',
                'age': 21
            },
            {
                'name': 'Carol'
            },
        ]
        ui.table(columns=columns, rows=rows, row_key='name')

    with ui.card():
        log = ui.log(max_lines=10).classes('w-full h-20')
        ui.button('Log time', on_click=lambda: log.push("log info here"))

    with ui.card():  # bind input

        class Demo:

            def __init__(self):
                self.number = 1

        demo = Demo()
        v = ui.checkbox('visible', value=True)
        with ui.column().bind_visibility_from(v, 'value'):
            ui.slider(min=1, max=3).bind_value(demo, 'number')
            ui.toggle({1: 'A', 2: 'B', 3: 'C'}).bind_value(demo, 'number')
            ui.number().bind_value(demo, 'number')

ui.run()
