"""Expression AST nodes for factor mining."""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, Union, List
from abc import ABC, abstractmethod
import json
import hashlib
from enum import Enum

class ExprNodeType(Enum):
    """Types of expression nodes."""
    ATOM = "atom"
    OPERATOR = "operator"
    CONSTANT = "constant"

@dataclass(frozen=True)
class ExprNode(ABC):
    """Base class for all expression nodes."""
    
    node_type: ExprNodeType
    domain: str = "pv_daily"
    frequency: str = "1d"
    depth: int = 0
    complexity: float = 1.0
    
    @abstractmethod
    def canonical_json(self) -> str:
        """Generate canonical JSON representation for hashing."""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Generate human-readable string representation."""
        pass
    
    @abstractmethod
    def get_children(self) -> List["ExprNode"]:
        """Get child nodes."""
        pass
    
    def hash(self) -> str:
        """Generate stable hash of the expression."""
        canonical = self.canonical_json()
        return hashlib.blake3(canonical.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_type": self.node_type.value,
            "domain": self.domain,
            "frequency": self.frequency,
            "depth": self.depth,
            "complexity": self.complexity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExprNode":
        """Create from dictionary."""
        node_type = ExprNodeType(data["node_type"])
        
        if node_type == ExprNodeType.ATOM:
            return ExprAtom.from_dict(data)
        elif node_type == ExprNodeType.OPERATOR:
            return ExprOperator.from_dict(data)
        else:
            raise ValueError(f"Unknown node type: {node_type}")

@dataclass(frozen=True)
class ExprAtom(ExprNode):
    """Atomic field expression node."""
    
    atom_name: str
    
    def __post_init__(self):
        object.__setattr__(self, 'node_type', ExprNodeType.ATOM)
    
    def canonical_json(self) -> str:
        """Generate canonical JSON for atom."""
        data = {
            "op": "atom",
            "name": self.atom_name,
            "domain": self.domain,
            "frequency": self.frequency
        }
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
    
    def to_string(self) -> str:
        """String representation of atom."""
        return f"{self.atom_name}"
    
    def get_children(self) -> List["ExprNode"]:
        """Atoms have no children."""
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert atom to dictionary."""
        data = super().to_dict()
        data.update({
            "atom_name": self.atom_name
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExprAtom":
        """Create atom from dictionary."""
        return cls(
            node_type=ExprNodeType.ATOM,
            domain=data["domain"],
            frequency=data["frequency"],
            depth=data.get("depth", 0),
            complexity=data.get("complexity", 1.0),
            atom_name=data["atom_name"]
        )

@dataclass(frozen=True)
class ExprOperator(ExprNode):
    """Operator expression node."""
    
    op_name: str
    args: Tuple["ExprNode", ...] = field(default_factory=tuple)
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        object.__setattr__(self, 'node_type', ExprNodeType.OPERATOR)
        
        # Calculate depth from children
        if self.args:
            max_child_depth = max(arg.depth for arg in self.args)
            object.__setattr__(self, 'depth', max_child_depth + 1)
    
    def canonical_json(self) -> str:
        """Generate canonical JSON for operator."""
        # Sort arguments for canonical representation (for commutative operators)
        if self.is_commutative():
            sorted_args = sorted(self.args, key=lambda x: x.canonical_json())
        else:
            sorted_args = self.args
        
        data = {
            "op": self.op_name,
            "domain": self.domain,
            "frequency": self.frequency,
            "args": [arg.canonical_json() for arg in sorted_args]
        }
        
        # Add parameters if any
        if self.params:
            # Sort parameters for canonical representation
            sorted_params = dict(sorted(self.params.items()))
            data["params"] = sorted_params
        
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
    
    def to_string(self) -> str:
        """String representation of operator."""
        if not self.args:
            return f"{self.op_name}()"
        
        arg_strings = [arg.to_string() for arg in self.args]
        
        if self.params:
            param_strings = [f"{k}={v}" for k, v in self.params.items()]
            return f"{self.op_name}({', '.join(arg_strings)}, {', '.join(param_strings)})"
        else:
            return f"{self.op_name}({', '.join(arg_strings)})"
    
    def get_children(self) -> List["ExprNode"]:
        """Get child nodes."""
        return list(self.args)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert operator to dictionary."""
        data = super().to_dict()
        data.update({
            "op_name": self.op_name,
            "args": [arg.to_dict() for arg in self.args],
            "params": self.params
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExprOperator":
        """Create operator from dictionary."""
        args = tuple(ExprNode.from_dict(arg_data) for arg_data in data["args"])
        
        return cls(
            node_type=ExprNodeType.OPERATOR,
            domain=data["domain"],
            frequency=data["frequency"],
            depth=data.get("depth", 0),
            complexity=data.get("complexity", 1.0),
            op_name=data["op_name"],
            args=args,
            params=data.get("params", {})
        )
    
    def is_commutative(self) -> bool:
        """Check if operator is commutative."""
        commutative_ops = {"add", "mul", "and", "or", "min", "max"}
        return self.op_name in commutative_ops
    
    def get_complexity_score(self) -> float:
        """Calculate complexity score for this operator node."""
        base_complexity = {
            "atom": 1.0,
            "rank": 2.0,
            "zscore": 2.0,
            "ts_mean": 3.0,
            "ts_std": 3.0,
            "ts_rank": 3.0,
            "delta": 2.0,
            "decay_linear": 4.0,
            "add": 1.0,
            "sub": 1.0,
            "mul": 1.0,
            "safe_div": 2.0,
            "winsorize": 2.0,
            "log1p_abs": 2.0,
            "abs": 1.0,
            "sign": 1.0
        }
        
        base_score = base_complexity.get(self.op_name, 2.0)
        
        # Add child complexity
        child_score = sum(arg.get_complexity_score() for arg in self.args)
        
        # Add parameter complexity
        param_score = len(self.params) * 0.1
        
        return base_score + child_score + param_score

# Helper functions for creating common expressions
def atom(name: str, domain: str = "pv_daily", frequency: str = "1d") -> ExprAtom:
    """Create an atomic field expression."""
    return ExprAtom(
        atom_name=name,
        domain=domain,
        frequency=frequency,
        complexity=1.0
    )

def rank(expr: ExprNode) -> ExprOperator:
    """Create a rank operator expression."""
    return ExprOperator(
        op_name="rank",
        args=(expr,),
        domain=expr.domain,
        frequency=expr.frequency,
        complexity=expr.complexity + 2.0
    )

def ts_mean(expr: ExprNode, window: int) -> ExprOperator:
    """Create a time-series mean operator expression."""
    return ExprOperator(
        op_name="ts_mean",
        args=(expr,),
        params={"window": window},
        domain=expr.domain,
        frequency=expr.frequency,
        complexity=expr.complexity + 3.0
    )

def safe_div(left: ExprNode, right: ExprNode, eps: float = 1e-6) -> ExprOperator:
    """Create a safe division operator expression."""
    return ExprOperator(
        op_name="safe_div",
        args=(left, right),
        params={"eps": eps},
        domain=left.domain,
        frequency=left.frequency,
        complexity=left.complexity + right.complexity + 2.0
    )

def add(left: ExprNode, right: ExprNode) -> ExprOperator:
    """Create an addition operator expression."""
    return ExprOperator(
        op_name="add",
        args=(left, right),
        domain=left.domain,
        frequency=left.frequency,
        complexity=left.complexity + right.complexity + 1.0
    )

def sub(left: ExprNode, right: ExprNode) -> ExprOperator:
    """Create a subtraction operator expression."""
    return ExprOperator(
        op_name="sub",
        args=(left, right),
        domain=left.domain,
        frequency=left.frequency,
        complexity=left.complexity + right.complexity + 1.0
    )

def mul(left: ExprNode, right: ExprNode) -> ExprOperator:
    """Create a multiplication operator expression."""
    return ExprOperator(
        op_name="mul",
        args=(left, right),
        domain=left.domain,
        frequency=left.frequency,
        complexity=left.complexity + right.complexity + 1.0
    )