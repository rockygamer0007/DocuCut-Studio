from __future__ import annotations

import sys
from pathlib import Path
import os
import ctypes

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

from docucut.engine.crop_engine import CropEngine
from docucut.engine.finalizer import finalize_smart
from docucut.engine.smart_detector import find_card_boxes_in_image
from docucut.engine.pdf_engine import (
    PDFEngine,
    PDFInvalidPassword,
    PDFPasswordRequired,
)
from docucut.profiles.store import ProfileStore


APP_NAME = "DocuCut Studio"
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = ROOT / "output" / "processed"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_NAME)
        icon_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "assets" / "docucut_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1400, 850)
        self.setMinimumSize(1100, 700)

        self.profile_store = ProfileStore()
        self.crop_engine = CropEngine()
        self.pdf_engine = PDFEngine()
        self.files: list[Path] = []
        self.last_outputs: list[Path] = []
        self.output_root = OUTPUT_ROOT
        self._detection_passwords: dict[Path, str] = {}

        self.build_ui()
        self.apply_theme()

    def build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 24, 18, 24)
        sidebar_layout.setSpacing(8)

        logo = QLabel(APP_NAME)
        logo.setObjectName("logo")

        subtitle = QLabel("Document crop & processing")
        subtitle.setObjectName("subtitle")

        sidebar_layout.addWidget(logo)
        sidebar_layout.addWidget(subtitle)
        sidebar_layout.addSpacing(28)

        for text in [
            "Workspace",
            "Files",
            "Crop Editor",
            "Profiles",
            "History",
        ]:
            button = QPushButton(text)
            button.setObjectName("navButton")
            sidebar_layout.addWidget(button)

        sidebar_layout.addStretch()

        settings_button = QPushButton("Settings")
        settings_button.setObjectName("navButton")
        sidebar_layout.addWidget(settings_button)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(18)

        header = QHBoxLayout()

        title_box = QVBoxLayout()

        title = QLabel("Workspace")
        title.setObjectName("pageTitle")

        description = QLabel(
            "Process PDFs and images using document-specific crop profiles."
        )
        description.setObjectName("pageDescription")

        title_box.addWidget(title)
        title_box.addWidget(description)

        header.addLayout(title_box)
        header.addStretch()

        add_files = QPushButton("+  Add Files")
        add_folder = QPushButton("+  Add Folder")

        add_files.setObjectName("primaryButton")
        add_folder.setObjectName("secondaryButton")

        add_files.clicked.connect(self.add_files)
        add_folder.clicked.connect(self.add_folder)

        header.addWidget(add_files)
        header.addWidget(add_folder)

        main_layout.addLayout(header)

        controls = QFrame()
        controls.setObjectName("card")

        controls_layout = QHBoxLayout(controls)

        profile_label = QLabel("Profile")
        self.profile_combo = QComboBox()

        for profile in self.profile_store.all():
            if profile.enabled:
                self.profile_combo.addItem(
                    f"{profile.id} — {profile.label}",
                    profile.id,
                )

        mode_label = QLabel("Output")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Fast · 300 DPI", False)
        self.mode_combo.addItem("PVC · 600 DPI", True)

        ai_label = QLabel("Detection")
        self.ai_combo = QComboBox()
        self.ai_combo.addItem("AI AUTO DETECT", True)
        self.ai_combo.addItem("Profile Crop", False)
        self.ai_combo.currentIndexChanged.connect(
            self.update_profile_control
        )

        controls_layout.addWidget(profile_label)
        controls_layout.addWidget(self.profile_combo, 1)

        controls_layout.addWidget(mode_label)
        controls_layout.addWidget(self.mode_combo)

        controls_layout.addWidget(ai_label)
        controls_layout.addWidget(self.ai_combo)

        main_layout.addWidget(controls)

        splitter = QSplitter(Qt.Horizontal)

        file_panel = QFrame()
        file_panel.setObjectName("card")

        file_layout = QVBoxLayout(file_panel)

        file_title = QLabel("Files")
        file_title.setObjectName("sectionTitle")

        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self.preview_selected)

        self.update_profile_control()

        queue_actions = QHBoxLayout()

        select_all_button = QPushButton("Select All")
        deselect_all_button = QPushButton("Deselect All")
        delete_selected_button = QPushButton("Delete Selected")

        for button in (
            select_all_button,
            deselect_all_button,
            delete_selected_button,
        ):
            button.setObjectName("secondaryButton")

        select_all_button.clicked.connect(self.select_all_files)
        deselect_all_button.clicked.connect(self.deselect_all_files)
        delete_selected_button.clicked.connect(self.delete_selected_files)

        queue_actions.addWidget(select_all_button)
        queue_actions.addWidget(deselect_all_button)
        queue_actions.addWidget(delete_selected_button)
        queue_actions.addStretch()

        file_layout.addWidget(file_title)
        file_layout.addLayout(queue_actions)
        file_layout.addWidget(self.file_list)

        preview_panel = QFrame()
        preview_panel.setObjectName("card")

        preview_layout = QVBoxLayout(preview_panel)

        preview_title = QLabel("Preview")
        preview_title.setObjectName("sectionTitle")

        self.preview = QLabel("Select a file to preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setObjectName("preview")
        self.preview.setMinimumSize(400, 400)

        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.preview)

        splitter.addWidget(file_panel)
        splitter.addWidget(preview_panel)
        splitter.setSizes([500, 800])

        main_layout.addWidget(splitter, 1)

        status_bar = QHBoxLayout()

        self.status_label = QLabel(
            "0 files    •    0 processed    •    0 failed"
        )
        self.status_label.setObjectName("statusLabel")

        status_bar.addWidget(self.status_label)
        status_bar.addStretch()

        output_actions = QHBoxLayout()

        open_output_button = QPushButton("Open Output")
        browse_output_button = QPushButton("Browse Output")
        print_current_button = QPushButton("Print Current")
        print_all_button = QPushButton("Print All")

        for button in (
            open_output_button,
            browse_output_button,
            print_current_button,
            print_all_button,
        ):
            button.setObjectName("secondaryButton")

        open_output_button.clicked.connect(self.open_output)
        browse_output_button.clicked.connect(self.browse_output)
        print_current_button.clicked.connect(self.print_current)
        print_all_button.clicked.connect(self.print_all)

        output_actions.addWidget(open_output_button)
        output_actions.addWidget(browse_output_button)
        output_actions.addWidget(print_current_button)
        output_actions.addWidget(print_all_button)
        output_actions.addStretch()

        main_layout.addLayout(output_actions)

        retry_button = QPushButton("Retry Failed")
        retry_button.setObjectName("secondaryButton")

        process_button = QPushButton("Process Files")
        process_button.setObjectName("primaryButton")

        retry_button.clicked.connect(self.process_files)
        process_button.clicked.connect(self.process_files)

        status_bar.addWidget(retry_button)
        status_bar.addWidget(process_button)

        main_layout.addLayout(status_bar)

        root.addWidget(sidebar)
        root.addWidget(main)

        self.setCentralWidget(central)

    def update_profile_control(self):
        """Keep the Profile control consistent with Detection mode."""

        auto_detect = bool(self.ai_combo.currentData())

        self.profile_combo.setEnabled(not auto_detect)

        if auto_detect:
            self.profile_combo.setToolTip(
                "AI AUTO DETECT selects the profile separately for each file."
            )
        else:
            self.profile_combo.setToolTip(
                "Manual profile used when Profile Crop is selected."
            )

    def add_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Documents",
            str(ROOT),
            "Documents (*.pdf *.png *.jpg *.jpeg)",
        )

        self.add_paths([Path(x) for x in filenames])

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            str(ROOT),
        )

        if not folder:
            return

        paths = [
            p
            for p in Path(folder).rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
        ]

        self.add_paths(paths)

    def add_paths(self, paths: list[Path]):
        existing = set(self.files)

        for path in paths:
            if path.is_file() and path not in existing:
                self.files.append(path)

        self.refresh_file_list()

    def _queue_profile_label(self, source: Path) -> str:
        """Return a safe, non-interactive profile label for the queue."""

        name = source.name.casefold()

        # Strong filename evidence.
        for profile in self.profile_store.all():
            if not profile.enabled:
                continue

            for term in profile.detect_filename:
                token = str(term).casefold().strip()

                if token and token in name:
                    return profile.id

        # For PDFs, use only evidence available without prompting for
        # a password. Once a password has already been cached, use the
        # main detector so the queue label exactly matches processing.
        if source.suffix.lower() == ".pdf":
            if source in self._detection_passwords:
                try:
                    return self._detect_profile(source)[0].id
                except Exception:
                    pass

            try:
                password = self._detection_passwords.get(source)

                document = self.pdf_engine.open(
                    source,
                    password=password,
                )

                try:
                    page_count = len(document)

                    if page_count:
                        page = document[0]
                        width = float(page.rect.width)
                        height = float(page.rect.height)
                        short_edge = min(width, height)
                        long_edge = max(width, height)

                        if (
                            page_count >= 2
                            and 250.0 <= long_edge <= 400.0
                            and short_edge > 0
                            and 1.8 <= long_edge / short_edge <= 2.2
                        ):
                            return "AYUSHMAN"

                        page_text = (
                            page.get_text("text") or ""
                        ).casefold()

                        for profile in self.profile_store.all():
                            if not profile.enabled:
                                continue

                            for term in profile.detect_keywords:
                                keyword = (
                                    str(term)
                                    .casefold()
                                    .strip()
                                )

                                if keyword and keyword in page_text:
                                    return profile.id

                finally:
                    document.close()

            except (PDFPasswordRequired, PDFInvalidPassword):
                pass
            except Exception:
                pass

        return "AUTO"

    def refresh_file_list(self):
        self.file_list.clear()

        for path in self.files:
            profile_label = self._queue_profile_label(path)

            item = QListWidgetItem(
                f"{path.name}\n"
                f"    {profile_label}    ?    Ready"
            )

            item.setData(Qt.UserRole, str(path))
            item.setData(Qt.UserRole + 1, profile_label)
            item.setData(Qt.UserRole + 2, "Ready")

            item.setFlags(
                item.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsSelectable
                | Qt.ItemIsEnabled
            )

            item.setCheckState(Qt.Checked)
            self.file_list.addItem(item)

        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)
        else:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Select a file to preview")

        self.update_status(
            processed=0,
            failed=0,
        )

    def set_queue_status(self, source: Path, status: str):
        """Update the visible status of one queued file."""

        for index in range(self.file_list.count()):
            item = self.file_list.item(index)
            item_path = Path(item.data(Qt.UserRole))

            if item_path == source:
                profile_label = item.data(Qt.UserRole + 1) or "AUTO"
                item.setData(Qt.UserRole + 2, status)
                item.setText(
                    f"{source.name}\n"
                    f"    {profile_label}    ?    {status}"
                )
                self.file_list.viewport().update()
                return

    def select_all_files(self):
        for index in range(self.file_list.count()):
            self.file_list.item(index).setCheckState(Qt.Checked)

    def deselect_all_files(self):
        for index in range(self.file_list.count()):
            self.file_list.item(index).setCheckState(Qt.Unchecked)

    def delete_selected_files(self):
        checked_paths = set()

        for index in range(self.file_list.count()):
            item = self.file_list.item(index)
            if item.checkState() == Qt.Checked:
                checked_paths.add(Path(item.data(Qt.UserRole)))

        if not checked_paths:
            return

        self.files = [
            path for path in self.files
            if path not in checked_paths
        ]

        self.refresh_file_list()

    def preview_selected(self):
        item = self.file_list.currentItem()

        if not item:
            self.preview.setText("Select a file to preview")
            self.preview.setPixmap(QPixmap())
            return

        path = Path(item.data(Qt.UserRole))

        try:
            if path.suffix.lower() != ".pdf":
                self.show_image(path)
                return

            document = self.pdf_engine.open(path)

            try:
                profile, _ = self._detect_profile(path)
                layout = (profile.detect_layout or "").casefold()
                dpi = 120

                images = []

                # -----------------------------------------------------
                # ABHA: preview the actual front/back card regions.
                # -----------------------------------------------------
                if layout == "abha_vertical":
                    boxes = (
                        ("FRONT", profile.front),
                        ("BACK", profile.back),
                    )

                    for label, box in boxes:
                        if box is None:
                            continue

                        pixmap = self.pdf_engine.render_clip(
                            document,
                            (box.x0, box.y0, box.x1, box.y1),
                            page_number=0,
                            dpi=dpi,
                        )

                        temp = (
                            ROOT
                            / "output"
                            / "_runtime"
                            / f"preview_abha_{label.lower()}.png"
                        )
                        temp.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        pixmap.save(str(temp))

                        image = cv2.imread(
                            str(temp),
                            cv2.IMREAD_COLOR,
                        )

                        if image is not None:
                            images.append(image)

                # -----------------------------------------------------
                # Ayushman: preview page 1 and page 2.
                # -----------------------------------------------------
                elif layout == "page_per_side":
                    for page_number in range(min(2, len(document))):
                        pixmap = self.pdf_engine.render_page(
                            document,
                            page_number=page_number,
                            dpi=dpi,
                        )

                        temp = (
                            ROOT
                            / "output"
                            / "_runtime"
                            / f"preview_ayushman_page_{page_number + 1}.png"
                        )
                        temp.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        pixmap.save(str(temp))

                        image = cv2.imread(
                            str(temp),
                            cv2.IMREAD_COLOR,
                        )

                        if image is not None:
                            images.append(image)

                # -----------------------------------------------------
                # Existing/default PDF preview: first page.
                # -----------------------------------------------------
                else:
                    pixmap = self.pdf_engine.render_page(
                        document,
                        page_number=0,
                        dpi=dpi,
                    )

                    temp = (
                        ROOT
                        / "output"
                        / "_runtime"
                        / "preview_temp.png"
                    )
                    temp.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    pixmap.save(str(temp))

                    image = cv2.imread(
                        str(temp),
                        cv2.IMREAD_COLOR,
                    )

                    if image is not None:
                        images.append(image)

            finally:
                document.close()

            if not images:
                raise ValueError("Unable to render preview.")

            # ---------------------------------------------------------
            # Combine multiple preview images vertically.
            # Preserve each image's aspect ratio and center it.
            # ---------------------------------------------------------
            if len(images) == 1:
                preview_image = images[0]
            else:
                gap = 16
                width = max(image.shape[1] for image in images)
                height = (
                    sum(image.shape[0] for image in images)
                    + gap * (len(images) - 1)
                )

                preview_image = np.full(
                    (height, width, 3),
                    255,
                    dtype=np.uint8,
                )

                y = 0

                for index, image in enumerate(images):
                    x = (width - image.shape[1]) // 2

                    preview_image[
                        y:y + image.shape[0],
                        x:x + image.shape[1],
                    ] = image

                    y += image.shape[0]

                    if index < len(images) - 1:
                        y += gap

            preview_path = (
                ROOT
                / "output"
                / "_runtime"
                / "preview_combined.png"
            )

            cv2.imwrite(
                str(preview_path),
                preview_image,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )

            self.show_image(preview_path)

        except Exception as exc:
            self.preview.setText(
                f"Preview unavailable\n\n{exc}"
            )

    def show_image(self, path: Path):
        pixmap = QPixmap(str(path))

        if pixmap.isNull():
            self.preview.setText("Unable to preview image")
            return

        scaled = pixmap.scaled(
            self.preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.preview.setPixmap(scaled)

    def _detect_profile(self, source: Path):
        """Detect the document profile for one input file.

        Uses filename/text signals plus document-structure signals for
        specialized layouts such as Ayushman.
        Returns (Profile, password_used_or_None).
        """
        profiles = [
            profile
            for profile in self.profile_store.all()
            if profile.enabled
        ]

        filename_text = source.name.casefold()

        # -------------------------------------------------------------
        # Filename evidence.
        # -------------------------------------------------------------
        filename_scores = {}

        for profile in profiles:
            score = 0

            for term in profile.detect_filename:
                token = str(term).casefold().strip()

                if token and token in filename_text:
                    score += 3

            filename_scores[profile.id] = score

        # Non-PDF files only have filename evidence at this stage.
        if source.suffix.lower() != ".pdf":
            best = max(
                profiles,
                key=lambda item: filename_scores[item.id],
            )
            best_score = filename_scores[best.id]

            if best_score > 0:
                return best, None

            return (
                self.profile_store.get(
                    self.profile_combo.currentData()
                ),
                None,
            )

        # -------------------------------------------------------------
        # Open PDF and collect structural/text evidence.
        # -------------------------------------------------------------
        pdf_text = ""
        password_used = self._detection_passwords.get(source)

        while True:
            try:
                document = self.pdf_engine.open(
                    source,
                    password=password_used,
                )
                break

            except PDFPasswordRequired:
                password_value, ok = QInputDialog.getText(
                    self,
                    "PDF Password",
                    f"Enter password for:\n{source.name}",
                )

                if not ok:
                    raise

                password_used = password_value
                self._detection_passwords[source] = password_used
                self.refresh_file_list()

            except PDFInvalidPassword:
                self._detection_passwords.pop(source, None)

                password_value, ok = QInputDialog.getText(
                    self,
                    "PDF Password",
                    f"Incorrect password. Enter password for:\n{source.name}",
                )

                if not ok:
                    raise

                password_used = password_value
                self._detection_passwords[source] = password_used
                self.refresh_file_list()

        try:
            page_count = len(document)

            page_width = 0.0
            page_height = 0.0
            page_aspect = 0.0

            if page_count:
                first_page = document[0]

                pdf_text = first_page.get_text("text") or ""

                page_width = float(first_page.rect.width)
                page_height = float(first_page.rect.height)

                short_edge = min(page_width, page_height)
                long_edge = max(page_width, page_height)

                if short_edge > 0:
                    page_aspect = long_edge / short_edge

        finally:
            document.close()

        searchable_text = pdf_text.casefold()

        # -------------------------------------------------------------
        # Filename + searchable-text evidence.
        # -------------------------------------------------------------
        scores = dict(filename_scores)

        for profile in profiles:
            score = scores[profile.id]

            for term in profile.detect_keywords:
                keyword = str(term).casefold().strip()

                if not keyword:
                    continue

                if keyword in searchable_text:
                    score += 5

            if (
                profile.detect_min_page_w is not None
                and page_width >= profile.detect_min_page_w
            ):
                score += 1

            scores[profile.id] = score

        # -------------------------------------------------------------
        # Strong Aadhaar-specific evidence.
        #
        # Aadhaar documents can contain generic phrases such as
        # "Government of India" or "digitally signed", which overlap
        # with PAN detection keywords. Aadhaar-specific terms must
        # therefore receive a much stronger score.
        # -------------------------------------------------------------
        aadhaar = next(
            (
                profile
                for profile in profiles
                if profile.id.upper() == "AADHAAR"
            ),
            None,
        )

        if aadhaar is not None:
            aadhaar_signals = (
                "aadhaar",
                "aadhar",
                "uidai",
                "unique identification authority",
                "e-aadhaar",
                "vid",
            )

            strong_aadhaar_hits = sum(
                1
                for signal in aadhaar_signals
                if signal in searchable_text
            )

            if strong_aadhaar_hits:
                scores["AADHAAR"] = (
                    scores.get("AADHAAR", 0)
                    + strong_aadhaar_hits * 12
                )

        # -------------------------------------------------------------
        # Ayushman structural evidence.
        #
        # The original layout is page-per-side, so a two-page PDF with
        # the characteristic ~2:1 card-page geometry is strong evidence.
        # -------------------------------------------------------------
        ayushman = next(
            (
                profile
                for profile in profiles
                if profile.id.upper() == "AYUSHMAN"
            ),
            None,
        )

        if ayushman is not None:
            if (
                page_count >= 2
                and 250.0 <= max(page_width, page_height) <= 400.0
                and 1.8 <= page_aspect <= 2.2
            ):
                scores["AYUSHMAN"] = scores.get("AYUSHMAN", 0) + 8

        # -------------------------------------------------------------
        # ABHA structural/layout evidence.
        #
        # ABHA is a single vertical page, so don't confuse a generic
        # portrait document with ABHA solely from dimensions. Text or
        # filename evidence remains the primary signal.
        # -------------------------------------------------------------

        best = max(
            profiles,
            key=lambda item: scores[item.id],
        )
        best_score = scores[best.id]

        # Require actual evidence before claiming automatic detection.
        if best_score <= 0:
            return (
                self.profile_store.get(
                    self.profile_combo.currentData()
                ),
                password_used,
            )

        return best, password_used

    def process_files(self):
        if not self.files:
            QMessageBox.information(
                self,
                APP_NAME,
                "Add at least one PDF or image first.",
            )
            return

        manual_profile_id = self.profile_combo.currentData()

        pvc = bool(self.mode_combo.currentData())
        smart = bool(self.ai_combo.currentData())

        processed = 0
        failed = 0
        errors: list[str] = []

        selected_sources = [
            Path(self.file_list.item(index).data(Qt.UserRole))
            for index in range(self.file_list.count())
            if self.file_list.item(index).checkState() == Qt.Checked
        ]

        if not selected_sources:
            QMessageBox.information(
                self,
                APP_NAME,
                "Select at least one file to process.",
            )
            return

        for source in selected_sources:
            try:
                self.set_queue_status(source, "Processing")
                QApplication.processEvents()

                if smart:
                    profile, detection_password = self._detect_profile(source)
                else:
                    profile = self.profile_store.get(manual_profile_id)
                    detection_password = None

                target_width = (
                    profile.output_pvc_width
                    if pvc
                    else profile.output_standard_width
                )

                if target_width is None:
                    raise ValueError(
                        f"{profile.id} has no configured output width."
                    )

                if source.suffix.lower() == ".pdf":
                    outputs = self.process_pdf(
                        source,
                        profile,
                        smart,
                        target_width,
                        password=detection_password,
                    )
                else:
                    outputs = self.process_image(
                        source,
                        profile,
                        smart,
                        target_width,
                    )

                processed += len(outputs)
                self.last_outputs.extend(outputs)

                self.set_queue_status(source, "Completed")
                QApplication.processEvents()

            except Exception as exc:
                failed += 1
                errors.append(f"{source.name}: {exc}")

                self.set_queue_status(source, "Failed")
                QApplication.processEvents()

        self.status_label.setText(
            f"{len(self.files)} files    •    "
            f"{processed} processed    •    "
            f"{failed} failed"
        )

        if errors:
            QMessageBox.warning(
                self,
                APP_NAME,
                "Some files failed:\n\n"
                + "\n".join(errors[:10]),
            )
        else:
            QMessageBox.information(
                self,
                APP_NAME,
                f"Processing complete.\n\n"
                f"Created {processed} output file(s).",
            )

    def process_image(
        self,
        source: Path,
        profile,
        smart: bool,
        target_width: int,
    ) -> list[Path]:
        output_dir = self.output_root / profile.id

        results = []

        sides = ("front", "back") if profile.sides == "dual" else ("front",)

        for side in sides:
            output = output_dir / f"{source.stem}-{side.upper()}.png"

            result = self.crop_engine.process_image(
                source,
                profile,
                side,
                output,
                smart=smart,
                target_width=target_width,
            )

            results.append(result.output)

        return results

    def process_pdf(
        self,
        source: Path,
        profile,
        smart: bool,
        target_width: int,
        password: str | None = None,
    ) -> list[Path]:
        """Process a PDF according to the selected/detected profile layout."""

        # -------------------------------------------------------------
        # Open/authenticate the PDF.
        # -------------------------------------------------------------
        while True:
            try:
                document = self.pdf_engine.open(
                    source,
                    password=password,
                )
                break

            except PDFPasswordRequired:
                password_value, ok = QInputDialog.getText(
                    self,
                    "PDF Password",
                    f"Enter password for:\n{source.name}",
                )

                if not ok:
                    raise

                password = password_value

            except PDFInvalidPassword:
                password_value, ok = QInputDialog.getText(
                    self,
                    "PDF Password",
                    f"Incorrect password. Enter password for:\n{source.name}",
                )

                if not ok:
                    raise

                password = password_value

        try:
            dpi = 600 if target_width >= 2022 else 300
            layout = (profile.detect_layout or "").casefold()

            # ---------------------------------------------------------
            # 1. SMART / AI detection
            #
            # Aadhaar already has a specialized correction inside
            # find_card_boxes_in_image(..., profile_id="AADHAAR").
            # ---------------------------------------------------------
            if smart and layout == "smart":
                detect_dpi = 150

                detect_path = (
                    ROOT
                    / "output"
                    / "_runtime"
                    / f"{source.stem}_detect_150dpi.png"
                )
                detect_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                detect_pixmap = self.pdf_engine.render_page(
                    document,
                    page_number=0,
                    dpi=detect_dpi,
                )
                detect_pixmap.save(str(detect_path))

                detect_image = cv2.imread(
                    str(detect_path),
                    cv2.IMREAD_COLOR,
                )

                if detect_image is None:
                    raise ValueError(
                        f"Unable to read SMART detection render: {detect_path}"
                    )

                boxes = find_card_boxes_in_image(
                    detect_image,
                    profile_id=profile.id,
                )

                if not boxes:
                    raise ValueError(
                        f"SMART detection found no card boxes for {profile.id}"
                    )

                sides = (
                    ("front", "back")
                    if profile.sides == "dual"
                    else ("front",)
                )

                results: list[Path] = []

                for index, side in enumerate(sides):
                    if index >= len(boxes):
                        raise ValueError(
                            f"SMART detection found only "
                            f"{len(boxes)} card(s) for {profile.id}"
                        )

                    clip_pixmap = self.pdf_engine.render_clip(
                        document,
                        boxes[index],
                        page_number=0,
                        dpi=dpi,
                    )

                    clip_path = (
                        ROOT
                        / "output"
                        / "_runtime"
                        / f"{source.stem}_{side}_clip_{dpi}dpi.png"
                    )

                    clip_pixmap.save(str(clip_path))

                    cropped = cv2.imread(
                        str(clip_path),
                        cv2.IMREAD_COLOR,
                    )

                    if cropped is None:
                        raise ValueError(
                            f"Unable to read PDF clip: {clip_path}"
                        )

                    finalized = finalize_smart(
                        cropped,
                        target_width,
                    )

                    output_dir = self.output_root / profile.id
                    output = (
                        output_dir
                        / f"{source.stem}-{side.upper()}.png"
                    )

                    results.append(
                        self.crop_engine.save(
                            finalized,
                            output,
                        )
                    )

                return results

            # ---------------------------------------------------------
            # 2. Aadhaar profile crop
            #
            # Use the same verified SMART detector + Aadhaar-specific
            # bottom correction that produces the approved Aadhaar
            # 1011x638 / 2022x1275 framing.
            #
            # This branch is used even when "Profile Crop" is selected,
            # so manual AADHAAR processing remains visually identical
            # to the verified AI result.
            # ---------------------------------------------------------
            if layout == "aadhaar_bottom":
                detect_dpi = 150

                detect_path = (
                    ROOT
                    / "output"
                    / "_runtime"
                    / f"{source.stem}_aadhaar_detect_150dpi.png"
                )
                detect_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                detect_pixmap = self.pdf_engine.render_page(
                    document,
                    page_number=0,
                    dpi=detect_dpi,
                )
                detect_pixmap.save(str(detect_path))

                detect_image = cv2.imread(
                    str(detect_path),
                    cv2.IMREAD_COLOR,
                )

                if detect_image is None:
                    raise ValueError(
                        f"Unable to read Aadhaar detection render: {detect_path}"
                    )

                boxes = find_card_boxes_in_image(
                    detect_image,
                    profile_id="AADHAAR",
                )

                if len(boxes) < 2:
                    raise ValueError(
                        f"Aadhaar detection found only "
                        f"{len(boxes)} card region(s)"
                    )

                results: list[Path] = []

                for index, side in enumerate(("front", "back")):
                    clip_pixmap = self.pdf_engine.render_clip(
                        document,
                        boxes[index],
                        page_number=0,
                        dpi=dpi,
                    )

                    clip_path = (
                        ROOT
                        / "output"
                        / "_runtime"
                        / f"{source.stem}_{side}_aadhaar_clip_{dpi}dpi.png"
                    )

                    clip_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    clip_pixmap.save(str(clip_path))

                    cropped = cv2.imread(
                        str(clip_path),
                        cv2.IMREAD_COLOR,
                    )

                    if cropped is None:
                        raise ValueError(
                            f"Unable to read Aadhaar PDF clip: {clip_path}"
                        )

                    finalized = finalize_smart(
                        cropped,
                        target_width,
                    )

                    output_dir = self.output_root / profile.id
                    output = (
                        output_dir
                        / f"{source.stem}-{side.upper()}.png"
                    )

                    results.append(
                        self.crop_engine.save(
                            finalized,
                            output,
                        )
                    )

                return results

            # ---------------------------------------------------------
            # 3. ABHA vertical layout
            #
            # The ABHA profile is two vertically stacked regions on
            # the first page. These coordinates are the profile's
            # calibrated fallback regions.
            # ---------------------------------------------------------
            if layout == "abha_vertical":
                sides = (
                    ("front", profile.front),
                    ("back", profile.back),
                )

                results: list[Path] = []

                for side, box in sides:
                    if box is None:
                        continue

                    clip_pixmap = self.pdf_engine.render_clip(
                        document,
                        (box.x0, box.y0, box.x1, box.y1),
                        page_number=0,
                        dpi=dpi,
                    )

                    clip_path = (
                        ROOT
                        / "output"
                        / "_runtime"
                        / f"{source.stem}_{side}_abha_{dpi}dpi.png"
                    )

                    clip_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    clip_pixmap.save(str(clip_path))

                    cropped = cv2.imread(
                        str(clip_path),
                        cv2.IMREAD_COLOR,
                    )

                    if cropped is None:
                        raise ValueError(
                            f"Unable to read ABHA PDF clip: {clip_path}"
                        )

                    finalized = finalize_smart(
                        cropped,
                        target_width,
                    )

                    output_dir = self.output_root / profile.id
                    output = (
                        output_dir
                        / f"{source.stem}-{side.upper()}.png"
                    )

                    results.append(
                        self.crop_engine.save(
                            finalized,
                            output,
                        )
                    )

                if not results:
                    raise ValueError(
                        f"No ABHA crop regions configured for {source.name}"
                    )

                return results

            # ---------------------------------------------------------
            # 3. Ayushman page-per-side layout
            #
            # Page 1 is FRONT, page 2 is BACK.
            # The profile uses a near-full-page crop on each page.
            # ---------------------------------------------------------
            if layout == "page_per_side":
                page_count = len(document)

                if page_count < 1:
                    raise ValueError(
                        f"PDF contains no pages: {source.name}"
                    )

                sides = (
                    ("front", 0),
                    ("back", 1),
                )

                results: list[Path] = []

                for side, page_number in sides:
                    if page_number >= page_count:
                        # A one-page document produces only the side
                        # that actually exists.
                        continue

                    box = (
                        profile.front
                        if side == "front"
                        else profile.back
                    )

                    if box is None:
                        continue

                    clip_pixmap = self.pdf_engine.render_clip(
                        document,
                        (box.x0, box.y0, box.x1, box.y1),
                        page_number=page_number,
                        dpi=dpi,
                    )

                    clip_path = (
                        ROOT
                        / "output"
                        / "_runtime"
                        / (
                            f"{source.stem}_{side}"
                            f"_ayushman_{dpi}dpi.png"
                        )
                    )

                    clip_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    clip_pixmap.save(str(clip_path))

                    cropped = cv2.imread(
                        str(clip_path),
                        cv2.IMREAD_COLOR,
                    )

                    if cropped is None:
                        raise ValueError(
                            f"Unable to read Ayushman PDF clip: {clip_path}"
                        )

                    finalized = finalize_smart(
                        cropped,
                        target_width,
                    )

                    output_dir = self.output_root / profile.id
                    output = (
                        output_dir
                        / f"{source.stem}-{side.upper()}.png"
                    )

                    results.append(
                        self.crop_engine.save(
                            finalized,
                            output,
                        )
                    )

                if not results:
                    raise ValueError(
                        f"No Ayushman card pages found in {source.name}"
                    )

                return results

            # ---------------------------------------------------------
            # 4. Existing fixed/profile crop path.
            #
            # This intentionally remains page 1 only for profiles
            # that don't declare a specialized multi-page layout.
            # ---------------------------------------------------------
            render_path = (
                ROOT
                / "output"
                / "_runtime"
                / f"{source.stem}_render_{dpi}dpi.png"
            )

            render_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            pixmap = self.pdf_engine.render_page(
                document,
                page_number=0,
                dpi=dpi,
            )
            pixmap.save(str(render_path))

        finally:
            document.close()

        return self.process_image(
            render_path,
            profile,
            smart=False,
            target_width=target_width,
        )

    def open_output(self):
        folder = OUTPUT_ROOT

        if not folder.is_dir():
            folder.mkdir(parents=True, exist_ok=True)

        os.startfile(str(folder))

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(self.output_root),
        )

        if not folder:
            return

        global OUTPUT_ROOT
        OUTPUT_ROOT = Path(folder)

        QMessageBox.information(
            self,
            APP_NAME,
            f"Output folder set to:\n\n{OUTPUT_ROOT}",
        )

    def print_current(self):
        if not self.last_outputs:
            QMessageBox.information(
                self,
                APP_NAME,
                "There are no processed outputs to print.",
            )
            return

        path = self.last_outputs[-1]

        try:
            os.startfile(str(path), "print")
        except OSError as exc:
            QMessageBox.warning(
                self,
                APP_NAME,
                f"Could not print:\n{path.name}\n\n{exc}",
            )

    def print_all(self):
        if not self.last_outputs:
            QMessageBox.information(
                self,
                APP_NAME,
                "There are no processed outputs to print.",
            )
            return

        answer = QMessageBox.question(
            self,
            APP_NAME,
            f"Send {len(self.last_outputs)} PNG(s) to the printer?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        printed = 0
        errors = []

        for path in self.last_outputs:
            try:
                os.startfile(str(path), "print")
                printed += 1
            except OSError as exc:
                errors.append(f"{path.name}: {exc}")

        message = f"Sent {printed} file(s) to the printer."

        if errors:
            message += "\n\nFailed:\n" + "\n".join(errors[:5])

        QMessageBox.information(
            self,
            APP_NAME,
            message,
        )
    def update_status(self, processed: int, failed: int):
        self.status_label.setText(
            f"{len(self.files)} files    •    "
            f"{processed} processed    •    "
            f"{failed} failed"
        )

    def apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f5f7fb;
            }

            #sidebar {
                background: #101827;
            }

            #logo {
                color: white;
                font-size: 21px;
                font-weight: 700;
            }

            #subtitle {
                color: #9ca7b8;
                font-size: 11px;
            }

            #navButton {
                text-align: left;
                padding: 12px 14px;
                border: none;
                border-radius: 8px;
                color: #c8d1df;
                background: transparent;
                font-size: 13px;
            }

            #navButton:hover {
                background: #1c293b;
                color: white;
            }

            #pageTitle {
                font-size: 28px;
                font-weight: 700;
                color: #111827;
            }

            #pageDescription {
                color: #6b7280;
                font-size: 13px;
            }

            #card {
                background: white;
                border: 1px solid #e4e8ef;
                border-radius: 12px;
            }

            #sectionTitle {
                font-size: 15px;
                font-weight: 600;
                color: #1f2937;
            }

            #preview {
                color: #9ca3af;
                background: #f8fafc;
                border: 1px dashed #cbd5e1;
                border-radius: 8px;
            }

            #primaryButton {
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 18px;
                font-weight: 600;
            }

            #primaryButton:hover {
                background: #1d4ed8;
            }

            #secondaryButton {
                background: white;
                color: #374151;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 10px 18px;
            }

            #secondaryButton:hover {
                background: #f9fafb;
            }

            #statusLabel {
                color: #6b7280;
                font-size: 12px;
            }

            QLabel {
                color: #374151;
            }

            QComboBox {
                padding: 8px 10px;
                border: 1px solid #d1d5db;
                border-radius: 7px;
                background: white;
                color: #111827;
                min-width: 130px;
            }

            QComboBox QAbstractItemView {
                background: white;
                color: #111827;
                selection-background-color: #2563eb;
                selection-color: white;
                border: 1px solid #d1d5db;
            }

            QListWidget {
                border: none;
                background: transparent;
                color: #111827;
            }

            QListWidget::item {
                color: #111827;
                background: transparent;
                padding: 10px;
                border-radius: 6px;
            }

            QListWidget::item {
                color: #111827;
                background: transparent;
                padding: 10px;
                border-radius: 6px;
            }

            QListWidget::item:hover {
                background: #eff6ff;
                color: #111827;
            }

            QListWidget::item:selected {
                color: white;
                background: #2563eb;
            }

            QListWidget::item:selected:hover {
                color: white;
                background: #2563eb;
            }

            QListWidget::item:focus {
                outline: none;
                border: none;
            }

            QListWidget {
                outline: none;
            }

            QListWidget::indicator {
                width: 15px;
                height: 15px;
            }

            QListWidget::indicator:unchecked {
                border: 1px solid #9ca3af;
                border-radius: 4px;
                background: white;
            }

            QListWidget::indicator:checked {
                border: 1px solid #2563eb;
                border-radius: 4px;
                background: #2563eb;
            }
            """
        )


def main():
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DocuCutStudio")
    except Exception:
        pass
    app = QApplication(sys.argv)
    icon_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "assets" / "docucut_icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    app.setApplicationName(APP_NAME)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()







