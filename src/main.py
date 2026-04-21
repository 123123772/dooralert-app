import flet as ft
import cv2
import numpy as np
from pathlib import Path
import sys
import traceback
import tempfile
import uuid
import base64
import csv
import asyncio
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent))
from door_alert_system import DoorAlertSystem

executor = ThreadPoolExecutor(max_workers=2)

def main(page: ft.Page):
    page.title = "车门开门预警系统"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO
    page.window_width = 1200
    page.window_height = 800
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # 模型初始化
    model_path = "models/best.pt"
    config_path = Path(__file__).parent / "camera_config.yaml"
    if not Path(model_path).exists():
        page.add(ft.Text(f"错误: 模型文件不存在 {model_path}", color="red"))
        return
    alert_sys = DoorAlertSystem(str(model_path), str(config_path) if config_path.exists() else None)

    selected_files = []
    detection_results = {}

    # ----- 控件 -----
    car_conf_value = ft.Text("0.50", size=14, weight="bold")
    bike_conf_value = ft.Text("0.50", size=14, weight="bold")
    person_conf_value = ft.Text("0.30", size=14, weight="bold")

    def update_car_val(e): car_conf_value.value = f"{e.control.value:.2f}"; page.update()
    def update_bike_val(e): bike_conf_value.value = f"{e.control.value:.2f}"; page.update()
    def update_person_val(e): person_conf_value.value = f"{e.control.value:.2f}"; page.update()

    car_slider = ft.Slider(min=0.1, max=1.0, value=0.5, on_change=update_car_val)
    bike_slider = ft.Slider(min=0.1, max=1.0, value=0.5, on_change=update_bike_val)
    person_slider = ft.Slider(min=0.1, max=1.0, value=0.3, on_change=update_person_val)

    file_picker = ft.FilePicker(on_result=lambda e: on_files_selected(e))
    page.overlay.append(file_picker)

    log_text = ft.TextField(multiline=True, read_only=True, height=150, label="日志", bgcolor="#F5F5F5")

    grid_list = ft.ListView(expand=True, spacing=15, padding=10)

    detail_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("文件名", weight="bold", size=12)),
            ft.DataColumn(ft.Text("类别", weight="bold", size=12)),
            ft.DataColumn(ft.Text("距离(m)", weight="bold", size=12)),
            ft.DataColumn(ft.Text("预警级别", weight="bold", size=12)),
            ft.DataColumn(ft.Text("置信度", weight="bold", size=12)),
        ],
        rows=[],
        heading_row_color="#E0E0E0",
    )
    table_container = ft.Container(
        content=ft.Column([detail_table], scroll=ft.ScrollMode.AUTO),
        height=300,
        border=ft.border.all(1, "#CCCCCC"),
        border_radius=8,
        padding=5,
    )

    def add_log(msg):
        print(msg)
        log_text.value += f"{msg}\n"
        page.update()

    def on_files_selected(e: ft.FilePickerResultEvent):
        if e.files:
            selected_files.clear()
            selected_files.extend([f.path for f in e.files])
            add_log(f"已选择 {len(selected_files)} 个图片文件")
            detection_results.clear()
            refresh_grid()
            refresh_detail_table()
        else:
            add_log("未选择文件")

    def show_image_dialog(img_base64, detections):
        """弹出大图对话框，并显示该图片的检测结果列表"""
        add_log("尝试打开大图对话框")
        try:
            # 构建检测结果表格（仅当前图片）
            result_table = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("类别", weight="bold")),
                    ft.DataColumn(ft.Text("距离(m)", weight="bold")),
                    ft.DataColumn(ft.Text("预警级别", weight="bold")),
                    ft.DataColumn(ft.Text("置信度", weight="bold")),
                ],
                rows=[],
                heading_row_color="#E0E0E0",
            )
            for det in detections:
                level_str = "无" if det['warning_level'] == 0 else ("一级" if det['warning_level'] == 1 else "二级(紧急)")
                color = "red" if det['warning_level'] == 2 else ("orange" if det['warning_level'] == 1 else "green")
                result_table.rows.append(
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text(det['class_name'])),
                        ft.DataCell(ft.Text(str(det['distance']))),
                        ft.DataCell(ft.Text(level_str, color=color)),
                        ft.DataCell(ft.Text(f"{det['confidence']:.2f}")),
                    ])
                )
            dialog = ft.AlertDialog(
                title=ft.Text("检测结果详情"),
                content=ft.Container(
                    content=ft.Column([
                        ft.Image(src_base64=img_base64, width=800, height=500, fit=ft.ImageFit.CONTAIN),
                        ft.Divider(),
                        ft.Text("检测目标列表", weight="bold", size=14),
                        ft.Container(content=result_table, height=200, width=800),
                    ], spacing=10, scroll=ft.ScrollMode.AUTO),
                    width=850,
                    height=750,
                ),
                actions=[ft.TextButton("关闭", on_click=lambda e: page.close(dialog))],
            )
            page.open(dialog)
            add_log("大图对话框已打开")
        except Exception as e:
            add_log(f"打开大图失败: {str(e)}")

    def create_thumbnail_card(file_path, res):
        fname = Path(file_path).name
        if len(fname) > 40:
            fname = fname[:37] + "..."

        if res.get('error'):
            return ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Icon(name="error", size=50, color="red"),
                        ft.Text(fname, size=12, weight="bold", text_align="center", no_wrap=False),
                        ft.Text("处理失败", color="red", size=11),
                    ], spacing=8, alignment="center", horizontal_alignment="center"),
                    width=320,
                    padding=12,
                ),
            )
        elif res.get('img_base64'):
            current_img_base64 = res['img_base64']
            current_detections = res['detections']   # 获取检测目标列表
            img = ft.Image(src_base64=current_img_base64, width=280, height=180, fit=ft.ImageFit.CONTAIN)
            img_container = ft.Container(
                content=img,
                on_click=lambda e: show_image_dialog(current_img_base64, current_detections),
                ink=True,
                border_radius=10,
            )
            zoom_icon = ft.IconButton(
                icon="zoom_in",
                icon_size=24,
                tooltip="点击放大图片",
                on_click=lambda e: show_image_dialog(current_img_base64, current_detections),
            )
            stats = res['stats']
            return ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Stack([
                            img_container,
                            ft.Container(
                                content=zoom_icon,
                                alignment=ft.alignment.bottom_right,
                                margin=5,
                            ),
                        ]),
                        ft.Text(fname, size=11, weight="bold", text_align="center", no_wrap=False),
                        ft.Text(f"🚨紧急: {stats['emergency']}  ⚠️预警: {stats['warning']}  📏平均距离: {stats['avg_dist']:.1f}m",
                                size=10, text_align="center"),
                    ], spacing=6, alignment="center", horizontal_alignment="center"),
                    width=320,
                    padding=10,
                ),
            )
        else:
            return ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Icon(name="image", size=50, color="gray"),
                        ft.Text(fname, size=12, weight="bold", text_align="center", no_wrap=False),
                        ft.Text("未检测到目标", color="gray", size=11),
                    ], spacing=8, alignment="center", horizontal_alignment="center"),
                    width=320,
                    padding=12,
                ),
            )

    def refresh_grid():
        grid_list.controls.clear()
        row_controls = []
        cards_per_row = 2
        for i, (file_path, res) in enumerate(detection_results.items()):
            card = create_thumbnail_card(file_path, res)
            row_controls.append(card)
            if (i + 1) % cards_per_row == 0 or i == len(detection_results) - 1:
                grid_list.controls.append(
                    ft.Row(row_controls, alignment=ft.MainAxisAlignment.CENTER, spacing=20)
                )
                row_controls = []
        page.update()

    def refresh_detail_table():
        detail_table.rows.clear()
        for file_path, res in detection_results.items():
            fname = Path(file_path).name
            if res.get('error'):
                detail_table.rows.append(
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text(fname, size=11)),
                        ft.DataCell(ft.Text("处理失败", color="red", size=11)),
                        ft.DataCell(ft.Text("-", size=11)),
                        ft.DataCell(ft.Text("-", size=11)),
                        ft.DataCell(ft.Text("-", size=11)),
                    ])
                )
            elif not res.get('detections'):
                detail_table.rows.append(
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text(fname, size=11)),
                        ft.DataCell(ft.Text("无目标", color="gray", size=11)),
                        ft.DataCell(ft.Text("-", size=11)),
                        ft.DataCell(ft.Text("-", size=11)),
                        ft.DataCell(ft.Text("-", size=11)),
                    ])
                )
            else:
                for det in res['detections']:
                    level_str = "无" if det['warning_level'] == 0 else ("一级" if det['warning_level'] == 1 else "二级(紧急)")
                    color = "red" if det['warning_level'] == 2 else ("orange" if det['warning_level'] == 1 else "green")
                    detail_table.rows.append(
                        ft.DataRow(cells=[
                            ft.DataCell(ft.Text(fname, size=11)),
                            ft.DataCell(ft.Text(det['class_name'], size=11)),
                            ft.DataCell(ft.Text(str(det['distance']), size=11)),
                            ft.DataCell(ft.Text(level_str, color=color, size=11)),
                            ft.DataCell(ft.Text(f"{det['confidence']:.2f}", size=11)),
                        ])
                    )
        page.update()

    def process_single_image(file_path: str, conf_override: dict):
        img = cv2.imread(file_path)
        if img is None:
            return {'error': True, 'file_path': file_path, 'detections': [], 'stats': None, 'img_base64': None}
        try:
            annotated_img, detections = alert_sys.process_frame(img, conf_override)
            temp_dir = Path(tempfile.gettempdir()) / "door_alert_temp"
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / f"{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(temp_file), annotated_img)
            with open(temp_file, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
            emergency = sum(1 for d in detections if d['warning_level'] == 2)
            warning = sum(1 for d in detections if d['warning_level'] == 1)
            avg_dist = np.mean([d['distance'] for d in detections]) if detections else 0
            stats = {'emergency': emergency, 'warning': warning, 'avg_dist': avg_dist}
            return {'error': False, 'file_path': file_path, 'img_base64': img_base64, 'stats': stats, 'detections': detections}
        except Exception as e:
            add_log(f"处理 {Path(file_path).name} 出错: {str(e)}")
            return {'error': True, 'file_path': file_path, 'detections': [], 'stats': None, 'img_base64': None}

    async def run_batch_detection():
        if not selected_files:
            add_log("请先选择图片")
            return
        conf_override = {
            'car': car_slider.value,
            'cyclist': bike_slider.value,
            'person': person_slider.value
        }
        add_log(f"开始批量检测，共 {len(selected_files)} 张图片")
        detection_results.clear()
        refresh_grid()
        refresh_detail_table()
        for i, file_path in enumerate(selected_files):
            add_log(f"正在处理 [{i+1}/{len(selected_files)}]: {Path(file_path).name}")
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(executor, process_single_image, file_path, conf_override)
            if res:
                detection_results[file_path] = res
            else:
                detection_results[file_path] = {'error': True, 'file_path': file_path, 'detections': [], 'stats': None, 'img_base64': None}
            refresh_grid()
            refresh_detail_table()
            await asyncio.sleep(0.01)
        add_log("批量检测完成")

    def start_batch(e):
        def thread_start():
            asyncio.run(run_batch_detection())
        import threading
        threading.Thread(target=thread_start, daemon=True).start()
        add_log("批量检测已启动，请稍候...")

    def clear_all(e):
        selected_files.clear()
        detection_results.clear()
        refresh_grid()
        refresh_detail_table()
        add_log("已清空所有图片和结果")

    def export_csv(e):
        if not detection_results:
            add_log("无检测数据可导出")
            return
        rows = []
        for file_path, res in detection_results.items():
            fname = Path(file_path).name
            if res.get('error'):
                rows.append({"文件名": fname, "类别": "处理失败", "距离(m)": "", "预警级别": "", "置信度": ""})
            elif not res.get('detections'):
                rows.append({"文件名": fname, "类别": "无目标", "距离(m)": "", "预警级别": "", "置信度": ""})
            else:
                for det in res['detections']:
                    level_str = "无" if det['warning_level'] == 0 else ("一级" if det['warning_level'] == 1 else "二级(紧急)")
                    rows.append({
                        "文件名": fname,
                        "类别": det['class_name'],
                        "距离(m)": det['distance'],
                        "预警级别": level_str,
                        "置信度": det['confidence']
                    })
        temp_csv = Path(tempfile.gettempdir()) / f"detection_export_{uuid.uuid4().hex}.csv"
        with open(temp_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=["文件名", "类别", "距离(m)", "预警级别", "置信度"])
            writer.writeheader()
            writer.writerows(rows)
        add_log(f"导出CSV成功: {temp_csv}")
        page.show_snack_bar(ft.SnackBar(content=ft.Text(f"导出成功，文件保存至 {temp_csv}"), open=True))

    # ----- 布局 -----
    left_panel = ft.Container(
        content=ft.Column([
            ft.Text("置信度阈值调节", size=18, weight="bold"),
            ft.Row([ft.Text("汽车置信度", width=100), car_slider, car_conf_value]),
            ft.Row([ft.Text("自行车置信度", width=100), bike_slider, bike_conf_value]),
            ft.Row([ft.Text("行人置信度", width=100), person_slider, person_conf_value]),
            ft.Divider(),
            ft.Text("图片选择", weight="bold"),
            ft.Row([
                ft.ElevatedButton(
                    content=ft.Row([ft.Icon(name="upload_file"), ft.Text("选择图片")]),
                    on_click=lambda _: file_picker.pick_files(allow_multiple=True),
                ),
                ft.ElevatedButton("清空所有", on_click=clear_all, color="white", bgcolor="red"),
            ]),
            ft.Divider(),
            ft.Row([
                ft.ElevatedButton("开始批量检测", on_click=start_batch, color="white", bgcolor="green"),
                ft.ElevatedButton("导出CSV", on_click=export_csv),
            ], spacing=10),
            ft.Divider(),
            ft.Text("检测日志", weight="bold"),
            log_text,
        ], spacing=15),
        padding=15,
        bgcolor="#FAFAFA",
        border_radius=10,
        width=380,
    )

    right_panel = ft.Container(
        content=ft.Column([
            ft.Text("检测结果展示", size=18, weight="bold"),
            ft.Container(content=grid_list, expand=True, border=ft.border.all(1, "#CCCCCC"), border_radius=8, padding=10),
        ], spacing=10),
        padding=15,
        expand=True,
    )

    bottom_panel = ft.Container(
        content=ft.Column([
            ft.Text("详细检测记录", size=16, weight="bold"),
            table_container,
        ], spacing=10),
        padding=15,
        bgcolor="#FAFAFA",
        border_radius=10,
    )

    main_row = ft.Row(
        [left_panel, ft.VerticalDivider(), right_panel],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.START,
        expand=True,
    )

    page.add(main_row, bottom_panel)
    add_log("系统已启动，选择图片后点击「开始批量检测」，点击图片或右下角放大镜即可放大查看详情")

ft.app(target=main)