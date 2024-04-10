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

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    bool InitLLM(LLMAttrType attr);

    void append_textedit(const char *p_str);

    void enable_controls(bool b_enable);

private slots:
    void on_btn_ask_clicked();
    void resizeEvent(QResizeEvent *event) override;

    void on_txt_msg_returnPressed();

private:
    Ui::MainWindow *ui;
    QVBoxLayout *m_vBoxLayout;
    QVector<QTextEdit *> m_textedit_input_vec, m_textedit_output_vec;
    LLM m_llm;
};
#endif // MAINWINDOW_H
