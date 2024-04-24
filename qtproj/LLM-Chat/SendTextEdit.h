#pragma once

#include <QTextEdit>
#include <QKeyEvent>

class SendTextEdit : public QTextEdit
{
    Q_OBJECT
Q_SIGNALS:
    void returnPressSignal();

public:
    SendTextEdit(QWidget *parent) : QTextEdit(parent) {}

protected:
    void keyReleaseEvent(QKeyEvent *e) override
    {
        if (e->key() == Qt::Key_Return)
        {
            emit returnPressSignal();
            return;
        }
    }
};
