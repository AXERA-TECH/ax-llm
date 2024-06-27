#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QHBoxLayout>
#include <QTextEdit>
#include "src/runner/LLM.hpp"

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

signals:
    void sig_llm_output(QString str);
    void sig_enable_controls(bool enable);

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    bool InitLLM(LLMAttrType attr);

    void append_textedit(const char *p_str);

    void enable_controls(bool b_enable);

private slots:
    void on_btn_stop_clicked();

private slots:
    void on_btn_ask_clicked();

    void on_llm_output(QString str);
    void on_enable_controls(bool b_enable);

    void on_horizontalSlider_valueChanged(int value);

private:
    Ui::MainWindow *ui;
    LLM m_llm;

public:
    std::vector<unsigned short> m_data_en, m_data_zh;
};
#endif // MAINWINDOW_H
