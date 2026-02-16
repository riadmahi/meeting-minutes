'use client';

import { useEffect, useState, useCallback } from 'react';
import {
  FirestoreTask,
  getMeetingTasks,
  updateTaskStatus,
  addTask,
  deleteTask,
} from '@/services/firestoreService';
import { CheckCircle2, Circle, Clock, ListTodo, Plus, Trash2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

interface MeetingTasksPanelProps {
  meetingId: string;
}

export function MeetingTasksPanel({ meetingId }: MeetingTasksPanelProps) {
  const [tasks, setTasks] = useState<FirestoreTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newTaskDescription, setNewTaskDescription] = useState('');
  const [newTaskAssignee, setNewTaskAssignee] = useState('');

  const fetchTasks = useCallback(async () => {
    try {
      const meetingTasks = await getMeetingTasks(meetingId);
      setTasks(meetingTasks);
    } catch (err) {
      console.error('Failed to fetch meeting tasks:', err);
    } finally {
      setLoading(false);
    }
  }, [meetingId]);

  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  const handleToggleStatus = async (task: FirestoreTask) => {
    const newStatus = task.status === 'completed' ? 'pending' : 'completed';
    try {
      await updateTaskStatus(meetingId, task.id, newStatus);
      setTasks((prev) =>
        prev.map((t) =>
          t.id === task.id ? { ...t, status: newStatus, updatedAt: new Date() } : t,
        ),
      );
    } catch (err) {
      console.error('Failed to update task:', err);
    }
  };

  const handleAddTask = async () => {
    const desc = newTaskDescription.trim();
    if (!desc) return;

    try {
      const taskId = await addTask(meetingId, {
        description: desc,
        assignee: newTaskAssignee.trim() || undefined,
        status: 'pending',
      });
      setTasks((prev) => [
        ...prev,
        {
          id: taskId,
          meetingId,
          description: desc,
          assignee: newTaskAssignee.trim() || undefined,
          status: 'pending',
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ]);
      setNewTaskDescription('');
      setNewTaskAssignee('');
      setShowAddForm(false);
    } catch (err) {
      console.error('Failed to add task:', err);
    }
  };

  const handleDeleteTask = async (taskId: string) => {
    try {
      await deleteTask(meetingId, taskId);
      setTasks((prev) => prev.filter((t) => t.id !== taskId));
    } catch (err) {
      console.error('Failed to delete task:', err);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddTask();
    } else if (e.key === 'Escape') {
      setShowAddForm(false);
      setNewTaskDescription('');
      setNewTaskAssignee('');
    }
  };

  const pendingCount = tasks.filter((t) => t.status !== 'completed').length;

  if (loading) return null;

  return (
    <div className="border-t border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <ListTodo className="w-4 h-4 text-blue-500" />
          <h3 className="text-sm font-semibold text-gray-700">
            Tâches
            {pendingCount > 0 && (
              <span className="ml-1 text-xs text-gray-400">({pendingCount})</span>
            )}
          </h3>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowAddForm(!showAddForm)}
          className="h-7 px-2"
        >
          {showAddForm ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
        </Button>
      </div>

      {/* Add task form */}
      {showAddForm && (
        <div className="mb-3 space-y-2 p-2 bg-gray-50 rounded-lg">
          <Input
            value={newTaskDescription}
            onChange={(e) => setNewTaskDescription(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Description de la tâche..."
            autoFocus
          />
          <div className="flex gap-2">
            <Input
              value={newTaskAssignee}
              onChange={(e) => setNewTaskAssignee(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Assignée (optionnel)"
              className="flex-1"
            />
            <Button size="sm" onClick={handleAddTask} disabled={!newTaskDescription.trim()}>
              Ajouter
            </Button>
          </div>
        </div>
      )}

      {/* Tasks list */}
      {tasks.length === 0 && !showAddForm ? (
        <p className="text-xs text-gray-400 text-center py-2">Aucune tâche extraite</p>
      ) : (
        <div className="space-y-1">
          {tasks.map((task) => (
            <div
              key={task.id}
              className={`flex items-start gap-2 px-2 py-1.5 rounded group ${
                task.status === 'completed' ? 'opacity-60' : ''
              }`}
            >
              <button
                onClick={() => handleToggleStatus(task)}
                className="mt-0.5 flex-shrink-0"
              >
                {task.status === 'completed' ? (
                  <CheckCircle2 className="w-4 h-4 text-green-500" />
                ) : task.status === 'in_progress' ? (
                  <Clock className="w-4 h-4 text-orange-500" />
                ) : (
                  <Circle className="w-4 h-4 text-gray-300 hover:text-gray-500" />
                )}
              </button>

              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm leading-tight ${
                    task.status === 'completed' ? 'line-through text-gray-400' : 'text-gray-700'
                  }`}
                >
                  {task.description}
                </p>
                {(task.assignee || task.dueDate) && (
                  <div className="flex gap-1.5 mt-0.5">
                    {task.assignee && (
                      <span className="text-[10px] text-gray-400">{task.assignee}</span>
                    )}
                    {task.dueDate && (
                      <span className={`text-[10px] ${
                        new Date(task.dueDate) < new Date() && task.status !== 'completed'
                          ? 'text-red-500'
                          : 'text-gray-400'
                      }`}>
                        {new Date(task.dueDate).toLocaleDateString('fr-FR')}
                      </span>
                    )}
                  </div>
                )}
              </div>

              <button
                onClick={() => handleDeleteTask(task.id)}
                className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 text-gray-300 hover:text-red-500"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
