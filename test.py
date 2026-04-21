import flet as ft

def main(page: ft.Page):
    page.add(ft.Text("Hello, Flet! 窗口出现了"))

ft.app(target=main)