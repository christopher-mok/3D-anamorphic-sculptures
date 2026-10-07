import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  /** Rendered instead of the children after an error. */
  fallback?: ReactNode;
  /** Changing this value clears the error state (e.g. a new URL). */
  resetKey?: unknown;
  /** Optional label for console diagnostics. */
  label?: string;
  children: ReactNode;
}

interface State {
  error: Error | null;
  key: unknown;
}

/**
 * Small error boundary, mainly for three.js loaders (useGLTF / useLoader / useTexture)
 * which throw when a URL 404s. Keeps one missing asset from breaking the scene.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, key: this.props.resetKey };

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    if (props.resetKey !== state.key) return { error: null, key: props.resetKey };
    return null;
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.warn(`[ErrorBoundary${this.props.label ? `:${this.props.label}` : ""}]`, error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error) return this.props.fallback ?? null;
    return this.props.children;
  }
}
